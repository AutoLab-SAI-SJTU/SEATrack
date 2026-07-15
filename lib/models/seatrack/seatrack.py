"""
Basic seatrack model.
"""
import hashlib
import math
import os
from typing import List
from lib.models.layers.timm_compat import to_2tuple
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
from lib.models.layers.attn import MergedLinear
from lib.models.seatrack.vit_ci import vit_base_patch16_224_ce
from lib.train.admin.logging_utils import get_train_logger
from lib.utils.box_ops import box_xyxy_to_cxcywh


class SEATrack(nn.Module):
    """ This is the base class for seatrackrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4) #(1, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


class SmokeSEATrack(nn.Module):
    """Tiny model used only by the smoke training config."""

    def __init__(self, feat_sz):
        super().__init__()
        self.box_head = nn.Identity()
        self.feat_sz = feat_sz
        self.lora_box = nn.Parameter(torch.tensor([0.5, 0.5, 0.25, 0.25], dtype=torch.float32))
        self.lora_score = nn.Parameter(torch.zeros(1, 1, feat_sz, feat_sz, dtype=torch.float32))

    def forward(self, template, search, ce_template_mask=None, ce_keep_rate=None, return_last_attn=False):
        batch_size = search.shape[0]
        pred_boxes = self.lora_box.sigmoid().view(1, 1, 4).expand(batch_size, 1, 4)
        score_map = self.lora_score.sigmoid().expand(batch_size, 1, self.feat_sz, self.feat_sz)
        return {
            'pred_boxes': pred_boxes,
            'score_map': score_map,
            'backbone_feat': score_map,
        }


def _validate_lifttrack_config(cfg):
    bilift_cfg = getattr(cfg.MODEL, "BILIFT", None)
    if not getattr(bilift_cfg, "ENABLED", False):
        return

    gra_cfg = getattr(cfg.MODEL, "GRA", None)
    conflicts = []
    if getattr(cfg.MODEL, "AMG_ENABLED", True):
        conflicts.append("AMG")
    if getattr(cfg.MODEL, "HMOE_ENABLED", True):
        conflicts.append("HMoE")
    if getattr(gra_cfg, "ENABLED", False):
        conflicts.append("GRA")
    if getattr(gra_cfg, "DIAGNOSTICS", False):
        conflicts.append("GRA diagnostics")
    if conflicts:
        raise ValueError(
            "LiftTrack BiLift is incompatible with: {}. Disable these modules explicitly."
            .format(", ".join(conflicts))
        )


def _reset_lora_parameters_by_name(model, seed):
    for module_name, module in model.named_modules():
        if not isinstance(module, MergedLinear) or not hasattr(module, "lora_A"):
            continue
        digest = hashlib.sha256("{}:{}".format(int(seed), module_name).encode("utf-8")).digest()
        module_seed = int.from_bytes(digest[:8], byteorder="big") % (2 ** 63 - 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(module_seed)
            nn.init.xavier_normal_(module.lora_A, gain=math.sqrt(2))
            nn.init.zeros_(module.lora_B)


def build_seatrack(cfg, training=True, settings=None):
    _validate_lifttrack_config(cfg)
    model_logger = get_train_logger(settings, "model")
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')  # use pretrained OSTrack as initialization
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'smoke':
        return SmokeSEATrack(feat_sz=cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE)

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        gra_cfg = getattr(cfg.MODEL, "GRA", None)
        bilift_cfg = getattr(cfg.MODEL, "BILIFT", None)
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                           template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                           new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                           amglora_rank = cfg.MODEL.AMGLORA_RANK,
                                           hmoe_rank = cfg.MODEL.HMOE_RANK,
                                           gra_enabled=getattr(gra_cfg, "ENABLED", False),
                                           gra_diagnostics=getattr(gra_cfg, "DIAGNOSTICS", False),
                                           gra_layers=getattr(gra_cfg, "LAYERS", None),
                                           gra_rgae_enabled=getattr(gra_cfg, "RGAE_ENABLED", True),
                                           gra_rho_min=getattr(gra_cfg, "RHO_MIN", 0.1),
                                           gra_detach_rho=getattr(gra_cfg, "DETACH_RHO", False),
                                           amg_enabled=getattr(cfg.MODEL, "AMG_ENABLED", True),
                                           hmoe_enabled=getattr(cfg.MODEL, "HMOE_ENABLED", True),
                                           bilift_enabled=getattr(bilift_cfg, "ENABLED", False),
                                           bilift_layers=getattr(bilift_cfg, "LAYERS", []),
                                           bilift_rank=getattr(bilift_cfg, "RANK", 8),
                                           bilift_dropout=getattr(bilift_cfg, "DROPOUT", 0.0),
                                           bilift_diagnostics=getattr(bilift_cfg, "DIAGNOSTICS", False),
                                           settings=settings,
                                           )
        hidden_dim = backbone.embed_dim

    else:
        raise NotImplementedError
    """For prompt no need, because we have OSTrack as initialization"""
    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim, settings=settings)

    model = SEATrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu", weights_only=False)
        # import ipdb; ipdb.set_trace()
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        model_logger.info("Loaded pretrained model from %s", cfg.MODEL.PRETRAIN_FILE)
        model_logger.info("Missing keys: %s", missing_keys)
        model_logger.info("Unexpected keys: %s", unexpected_keys)

    if getattr(cfg.MODEL, "DETERMINISTIC_LORA_INIT", False):
        experiment_seed = int(getattr(settings, "seed", 0))
        _reset_lora_parameters_by_name(model, seed=experiment_seed)
        model_logger.info("Reset named LoRA parameters with experiment seed %d", experiment_seed)

    return model
