# How to use vae_process_images.py  

Encode and reconstruct images 
```bash 
python vae_process_images.py LunarLander-v2_img_10_200 LunarLander_64 64 -e -d 
```

Don't reencode latent vectors, but make reconstructions 
```bash
python vae_process_images.py LunarLander-v2_img_10_200 LunarLander_64 64 -l LunarLander-v2_img_10_200_latent -r
```