# EM data processing cheat sheet
## Class 2D
On bagpuss use a single thread with the GPU, seems multiple thread are competing badly.
Is it the same on other system?
## image handler
relion_image_handler --i 5A1A.mrc --angpix 1 --rescale_angpix 0.425 --o 5A1A_scaled.mrc
relion_image_handler --i 5A1A_scale.mrc --new_box 256 --o 5A1A_scaled_boxed.mrc

mrc_resize() {
    relion_image_handler --i $1 --rescale_angpix $2 --o tmp.mrc
    relion_image_handler --i tmp.mrc --new_box $3 --o rescale_$1
    rm tmp.mrc
}

##Optimal run parameters
Class3D
2GPUs 3MPI 20threads

Class 2D vdam
1MPI 1Thread