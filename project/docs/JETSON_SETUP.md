# This is the guide to Set Up the NVIDIA Jetson Orin Nano
## 1. Operating System Installation
Flash the official Jetson Linux OS onto your microSD card by following the NVIDIA Quick Start Guide:
👉 NVIDIA Jetson Orin Nano Developer Kit User Guide:
https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html

⚠️ Important Firmware Note: Pay close attention to your Jetson's firmware version. The flashing process and compatibility differ significantly between versions older than Jetpack 6 (Jetson Linux 36.0) and newer versions.

## 2. Hardware Compatibility Warning (Storage)
⚠️ CRITICAL: The M.2 slot on the Jetson Orin Nano carrier board only supports NVMe (PCIe) SSDs.

Do NOT buy or use M.2 SATA SSDs (even though they fit mechanically, they will not be recognized by the UEFI/BIOS or the OS).

Recommendation: Use high-speed NVMe drives (e.g., Samsung 980, Crucial P3) to maximize performance for Large Language Models (LLMs) and Docker containers.

## 3. Transferring Data from SD Card to NVMe SSD
Running the system from an NVMe SSD drastically speeds up data retrieval times for neural networks and LLM weights.

Install the NVMe SSD into the Jetson while keeping the microSD card inserted.

Clone the system partition helper script from GitHub:


- git clone https://github.com/jetsonhacks/copyRootToSSD.git
- cd copyRootToSSD

Run the script to format the NVMe SSD and copy all data from the SD card automatically:
- ./copyRootToSSD.sh -d /dev/nvme0n1

⚠️ Warning: Do NOT remove the microSD card after the transfer. The Jetson’s UEFI uses the SD card's EFI partition as a "bootstrap" (bootloader) before handing total system control over to the fast NVMe SSD.

## 4. Adjusting the Partition Identity (/etc/fstab)
To ensure the Jetson permanently uses the SSD as its root directory (/) upon reboot, you must modify the filesystem table on your newly copied SSD:

Find your SSD partition's unique identifier (UUID):

- sudo blkid /dev/nvme0n1p1

Mount the SSD temporarily to edit its configuration file:

- sudo mkdir -p /mnt/ssd
- sudo mount /dev/nvme0n1p1 /mnt/ssd
- sudo nano /mnt/ssd/etc/fstab

Replace the old /dev/root or SD card assignment with your SSD's UUID, but NEVER delete the /boot/efi line, otherwise the system will brick:

UUID=your-ssd-uuid-here    /            ext4    defaults    0    1
UUID=4EA2-9257             /boot/efi    vfat    defaults    0    1
Unmount the SSD before rebooting:

-sudo umount /mnt/ssd

## 5. Expanding Swap Memory (Crucial for LLMs)
The Jetson Orin Nano features 8 GB of unified physical RAM shared between the CPU and GPU. To prevent CUDA Out of Memory (error 12) crashes when running heavy models (like DeepSeek or Whisper), you must expand the Swap space on your new SSD to at least 24 GB - 32 GB.

Run the following commands sequentially to delete the old swap and allocate a massive new swapfile:

# Disable current swap
- sudo swapoff -a

# Allocate 32 GB of space on the SSD (change 32G to 24G if using a smaller SSD)
- sudo fallocate -l 32G /swapfile

# Set strict root permissions
- sudo chmod 600 /swapfile

# Format the file as Linux Swap
- sudo mkswap /swapfile

# Activate the new swap file immediately
- sudo swapon /swapfile

To make the swap space persistent after every reboot, append this line to the very bottom of your /etc/fstab:

/swapfile    none    swap    sw    0    0

## 6. Automating RAM Cache Cleanup (Preventing CUDA Issues)
Linux uses free memory to cache files (buff/cache). On Jetson's Unified Memory architecture, this cache can block the NVIDIA driver from allocating RAM to the GPU, causing LLMs to fail. We automate a cleanup script every 30 minutes.

Create the cleanup script:

- nano ~/clear_cache.sh
Paste the following code into the script:

#!/bin/bash
echo "=== Cache-Cleanup started: $(date) ==="
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo "=== Cache cleared successfully! ==="
Make the script executable:

- chmod +x ~/clear_cache.sh

Add the script to the root crontab for automation:

- sudo crontab -e
Append this plain text line at the very bottom to run it every 30 minutes and log the output:

*/30 * * * * /home/nano/clear_cache.sh >> /var/log/clear_cache.log 2>&1

## 7. Post-Installation Verification
After restarting your Jetson (sudo reboot), verify that the migration was successful by using these commands:

Verify Storage Boot: df -h / (Should point to /dev/nvme0n1p1)

Verify Memory Setup: free -h (Should display 31Gi or 23Gi under the Swap section)
