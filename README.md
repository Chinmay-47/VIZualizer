# VIZualizer


##### VIZualizer is an educational tool to visualize and learn popular machine learning algorithms.



#### Quick Fix for 'RuntimeError: Requested MovieWriter (ffmpeg) not available'


- Download the OS Appropriate Binary from https://ffbinaries.com/downloads
- Place in any directory on your PC
- Run the below function once in python console:

        import os
        ffmpeg_path = "insert-path-to-ffmpeg-binary"
        os.environ["PATH"] += os.pathsep + ffmpeg_path
 
 - Running the above function will add the binary to OS environment path