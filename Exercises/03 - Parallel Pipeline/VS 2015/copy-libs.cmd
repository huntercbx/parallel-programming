echo "Copying OpenCV dlls"
xcopy /Y %1\x64\vc14\bin\opencv_world341.dll %3
xcopy /Y %1\x64\vc14\bin\opencv_world341d.dll %3

echo "Copying TBB dlls"
xcopy /Y %2\bin\intel64\vc14\tbb.dll %3
xcopy /Y %2\bin\intel64\vc14\tbb.pdb %3
xcopy /Y %2\bin\intel64\vc14\tbb_debug.dll %3
xcopy /Y %2\bin\intel64\vc14\tbb_debug.pdb %3
