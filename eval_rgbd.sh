cd Depthtrack_workspace
vot evaluate --workspace ./ rgbd
vot analysis --name rgbd
cd ..
cd VOT22RGBD_workspace
vot evaluate --workspace ./ rgbd
vot analysis --name rgbd
cd ..
