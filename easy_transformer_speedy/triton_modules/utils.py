def num_warps_from_block_size(block_size):
  return 4 if block_size < 2048 else 8 if block_size < 4096 else 16
