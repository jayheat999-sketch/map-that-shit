# map-that-shit

$ python train_torus_english.py
  import pynvml  # type: ignore[import]
  Device: xpu

  Building model...
    ASCII coords loaded: 94
    Compounds: 6970 accepted, 432 fragments, 597 duplicates, 0 no coords
    Source: unified_angular
    Models: Llama-3.1-70B, Qwen2.5-72B
    ρ: 0.2314
  EnglishVocabulary:
    Special tokens: 4
    ASCII chars:    95 (IDs 4..98)
    Compounds:      6970 (IDs 99..7068)
    Total vocab:    7069
    Coord dims:     17
    ConsensusEmbedding: 7069 tokens × 17-d coords → 256-d model

  Parameters: 8,144,559 total, 8,144,559 trainable

  Loading dataset...
  Loading TinyStories from ./data/tinystories.txt...
  Loaded 2119489 stories from local file

════════════════════════════════════════════════════════════
  TRAINING
════════════════════════════════════════════════════════════
  Steps: 0 → 50000
  Batch size: 16
  Seq len: 256
  LR: 0.0003
  Timesteps: 10
════════════════════════════════════════════════════════════

  step      0 | loss 8.8643 (avg 8.8643) | acc 0.0% (avg 0.0%) | lr 3.00e-04 | 0.3 step/s
  step    100 | loss 5.1583 (avg 6.1246) | acc 24.7% (avg 17.1%) | lr 3.00e-04 | 8.1 step/s
  step    200 | loss 4.1970 (avg 4.6595) | acc 24.2% (avg 25.2%) | lr 3.00e-04 | 9.3 step/s
  step    300 | loss 3.5318 (avg 3.8439) | acc 27.0% (avg 25.1%) | lr 3.00e-04 | 9.8 step/s
  step    400 | loss 2.8069 (avg 3.3059) | acc 25.9% (avg 25.2%) | lr 3.00e-04 | 10.0 step/s
  step    500 | loss 2.6082 (avg 2.9269) | acc 25.6% (avg 25.1%) | lr 3.00e-04 | 10.2 step/s
  step    600 | loss 2.8406 (avg 2.6764) | acc 24.5% (avg 25.1%) | lr 3.00e-04 | 10.3 step/s
  step    700 | loss 2.5179 (avg 2.5691) | acc 26.0% (avg 25.2%) | lr 3.00e-04 | 10.4 step/s
  step    800 | loss 2.3251 (avg 2.5265) | acc 25.6% (avg 25.3%) | lr 3.00e-04 | 10.4 step/s
  step    900 | loss 2.5803 (avg 2.5241) | acc 26.0% (avg 25.3%) | lr 3.00e-04 | 10.5 step/s
  step   1000 | loss 2.6794 (avg 2.4634) | acc 25.5% (avg 25.4%) | lr 3.00e-04 | 10.5 step/s
  step   1100 | loss 2.3338 (avg 2.4340) | acc 23.6% (avg 25.2%) | lr 3.00e-04 | 10.6 step/s
  step   1200 | loss 2.5540 (avg 2.3218) | acc 24.5% (avg 25.4%) | lr 3.00e-04 | 10.6 step/s
  step   1300 | loss 2.0164 (avg 2.3171) | acc 25.4% (avg 25.6%) | lr 3.00e-04 | 10.6 step/s
  step   1400 | loss 2.5807 (avg 2.3605) | acc 24.8% (avg 25.4%) | lr 3.00e-04 | 10.6 step/s
  step   1500 | loss 2.5073 (avg 2.3150) | acc 25.1% (avg 25.4%) | lr 3.00e-04 | 10.6 step/s
  step   1600 | loss 1.5583 (avg 2.2810) | acc 25.7% (avg 25.8%) | lr 3.00e-04 | 10.6 step/s
  step   1700 | loss 2.0901 (avg 2.2286) | acc 25.4% (avg 25.7%) | lr 3.00e-04 | 10.6 step/s
  step   1800 | loss 1.9606 (avg 2.2562) | acc 22.8% (avg 25.8%) | lr 3.00e-04 | 10.6 step/s
  step   1900 | loss 2.2967 (avg 2.2587) | acc 26.9% (avg 25.9%) | lr 3.00e-04 | 10.6 step/s
  step   2000 | loss 2.5848 (avg 2.2318) | acc 27.0% (avg 25.9%) | lr 3.00e-04 | 10.6 step/s
  ╔══ sample (seed=2000) ══╗
  ║ Once rrnT a eelovedu .r. nce e T    g r            u                  
  ║ collapse: 100%
  ╚══════════════════════════════╝
  step   2100 | loss 2.2201 (avg 2.2246) | acc 27.0% (avg 26.2%) | lr 3.00e-04 | 10.6 step/s
  step   2200 | loss 2.3946 (avg 2.1828) | acc 27.3% (avg 26.5%) | lr 3.00e-04 | 10.6 step/s
  step   2300 | loss 1.7270 (avg 2.1983) | acc 27.6% (avg 26.7%) | lr 3.00e-04 | 10.6 step/s
  step   2400 | loss 1.7739 (avg 2.1521) | acc 27.6% (avg 26.9%) | lr 3.00e-04 | 10.6 step/s
  step   2500 | loss 2.1827 (avg 2.1743) | acc 27.8% (avg 27.1%) | lr 3.00e-04 | 10.6 step/s
  step   2600 | loss 1.9844 (avg 2.1509) | acc 27.3% (avg 27.1%) | lr 3.00e-04 | 10.6 step/s
  step   2700 | loss 1.7923 (avg 2.2098) | acc 27.0% (avg 27.1%) | lr 3.00e-04 | 10.6 step/s
  step   2800 | loss 2.2181 (avg 2.1347) | acc 26.3% (avg 27.3%) | lr 3.00e-04 | 10.6 step/s
  step   2900 | loss 1.7602 (avg 2.1637) | acc 28.2% (avg 27.2%) | lr 3.00e-04 | 10.6 step/s
  step   3000 | loss 2.2721 (avg 2.0711) | acc 25.6% (avg 27.3%) | lr 3.00e-04 | 10.7 step/s
  step   3100 | loss 2.5916 (avg 2.1282) | acc 26.5% (avg 27.4%) | lr 3.00e-04 | 10.7 step/s
  step   3200 | loss 1.9540 (avg 2.0883) | acc 28.8% (avg 27.6%) | lr 3.00e-04 | 10.7 step/s
  step   3300 | loss 2.0014 (avg 2.1083) | acc 27.7% (avg 27.6%) | lr 3.00e-04 | 10.7 step/s
  step   3400 | loss 1.8135 (avg 2.1097) | acc 26.3% (avg 27.5%) | lr 3.00e-04 | 10.7 step/s
  step   3500 | loss 2.3696 (avg 2.1391) | acc 27.2% (avg 28.0%) | lr 3.00e-04 | 10.7 step/s
  step   3600 | loss 2.1300 (avg 2.1084) | acc 27.8% (avg 27.8%) | lr 3.00e-04 | 10.7 step/s
  step   3700 | loss 2.2831 (avg 2.0527) | acc 27.1% (avg 27.9%) | lr 3.00e-04 | 10.7 step/s
  step   3800 | loss 2.3211 (avg 2.0293) | acc 29.5% (avg 28.1%) | lr 3.00e-04 | 10.7 step/s
  step   3900 | loss 2.1703 (avg 2.0808) | acc 26.7% (avg 28.2%) | lr 3.00e-04 | 10.7 step/s
  step   4000 | loss 2.3309 (avg 2.0612) | acc 29.2% (avg 28.1%) | lr 3.00e-04 | 10.7 step/s
  ╔══ sample (seed=4000) ══╗
  ║ Once therpon   aime, there w a   thertle      '  uts      a S         
  ║ collapse: 100%
  ╚══════════════════════════════╝
  step   4100 | loss 2.3985 (avg 2.0721) | acc 27.8% (avg 28.0%) | lr 3.00e-04 | 10.7 step/s
  step   4200 | loss 1.5281 (avg 2.0443) | acc 27.4% (avg 28.2%) | lr 3.00e-04 | 10.7 step/s
  step   4300 | loss 2.3726 (avg 2.0696) | acc 29.0% (avg 28.0%) | lr 3.00e-04 | 10.7 step/s
  step   4400 | loss 2.0698 (avg 2.0511) | acc 28.4% (avg 28.2%) | lr 3.00e-04 | 10.7 step/s
  step   4500 | loss 2.1415 (avg 2.0358) | acc 25.5% (avg 28.1%) | lr 3.00e-04 | 10.7 step/s
  step   4600 | loss 1.9838 (avg 2.0440) | acc 26.8% (avg 27.9%) | lr 3.00e-04 | 10.7 step/s
  step   4700 | loss 1.5906 (avg 2.0714) | acc 28.9% (avg 28.1%) | lr 3.00e-04 | 10.7 step/s
  step   4800 | loss 1.8435 (avg 2.0082) | acc 29.5% (avg 28.4%) | lr 3.00e-04 | 10.7 step/s
  step   4900 | loss 2.0257 (avg 1.9988) | acc 27.3% (avg 28.1%) | lr 3.00e-04 | 10.7 step/s
  step   5000 | loss 1.8073 (avg 1.9866) | acc 29.2% (avg 28.2%) | lr 3.00e-04 | 10.7 step/s
  ✓ saved ./checkpoints/torus_english_step_5000.pt
  step   5100 | loss 2.0074 (avg 2.0280) | acc 30.2% (avg 28.3%) | lr 3.00e-04 | 10.7 step/s
  step   5200 | loss 2.2761 (avg 2.0027) | acc 29.1% (avg 28.5%) | lr 3.00e-04 | 10.7 step/s
  step   5300 | loss 2.0775 (avg 2.0318) | acc 28.3% (avg 28.2%) | lr 3.00e-04 | 10.7 step/s
  step   5400 | loss 1.8896 (avg 1.9768) | acc 28.8% (avg 28.2%) | lr 3.00e-04 | 10.7 step/s
  step   5500 | loss 1.8988 (avg 2.0241) | acc 29.9% (avg 28.4%) | lr 3.00e-04 | 10.7 step/s
  step   5600 | loss 2.0408 (avg 1.9944) | acc 29.2% (avg 28.6%) | lr 3.00e-04 | 10.7 step/s
  step   5700 | loss 2.3318 (avg 2.0091) | acc 28.9% (avg 28.5%) | lr 3.00e-04 | 10.7 step/s
  step   5800 | loss 2.3363 (avg 2.0164) | acc 28.9% (avg 28.6%) | lr 3.00e-04 | 10.7 step/s
  step   5900 | loss 2.2716 (avg 1.9937) | acc 30.0% (avg 28.3%) | lr 3.00e-04 | 10.7 step/s
  step   6000 | loss 1.6692 (avg 1.9739) | acc 30.9% (avg 28.4%) | lr 3.00e-04 | 10.8 step/s
  ╔══ sample (seed=6000) ══╗
  ║ Once upon a time, there w s a toptle    ins      .       h          d 
  ║ collapse: 100%
  ╚══════════════════════════════╝
  step   6100 | loss 2.1127 (avg 1.9866) | acc 27.4% (avg 28.4%) | lr 3.00e-04 | 10.7 step/s
  step   6200 | loss 1.8683 (avg 1.9772) | acc 28.4% (avg 28.6%) | lr 3.00e-04 | 10.7 step/s
  step   6300 | loss 2.0728 (avg 1.9899) | acc 28.0% (avg 28.5%) | lr 3.00e-04 | 10.7 step/s
  step   6400 | loss 2.1147 (avg 1.9409) | acc 28.7% (avg 28.8%) | lr 3.00e-04 | 10.7 step/s
  step   6500 | loss 2.3330 (avg 1.9740) | acc 28.8% (avg 28.7%) | lr 3.00e-04 | 10.7 step/s
  step   6600 | loss 2.3676 (avg 1.9760) | acc 30.4% (avg 28.6%) | lr 3.00e-04 | 10.7 step/s
  step   6700 | loss 1.7419 (avg 1.9736) | acc 27.6% (avg 28.4%) | lr 3.00e-04 | 10.7 step/s
  step   6800 | loss 2.2985 (avg 1.9938) | acc 25.3% (avg 28.7%) | lr 3.00e-04 | 10.7 step/s
  step   6900 | loss 2.1548 (avg 1.9466) | acc 29.5% (avg 28.6%) | lr 3.00e-04 | 10.7 step/s
  step   7000 | loss 1.8548 (avg 1.9864) | acc 29.1% (avg 28.7%) | lr 3.00e-04 | 10.7 step/s
  step   7100 | loss 2.2399 (avg 2.0063) | acc 30.6% (avg 28.6%) | lr 3.00e-04 | 10.7 step/s
  step   7200 | loss 2.4036 (avg 1.9895) | acc 29.4% (avg 28.9%) | lr 3.00e-04 | 10.7 step/s
  step   7300 | loss 2.1385 (avg 1.9763) | acc 29.9% (avg 28.5%) | lr 3.00e-04 | 10.7 step/s
  step   7400 | loss 2.2442 (avg 1.9820) | acc 28.7% (avg 28.7%) | lr 3.00e-04 | 10.7 step/s
  step   7500 | loss 1.9468 (avg 1.9555) | acc 26.8% (avg 28.8%) | lr 3.00e-04 | 10.7 step/s
  step   7600 | loss 2.0951 (avg 1.9590) | acc 27.3% (avg 28.6%) | lr 3.00e-04 | 10.7 step/s
  step   7700 | loss 2.0728 (avg 1.9808) | acc 28.6% (avg 28.5%) | lr 3.00e-04 | 10.7 step/s
  step   7800 | loss 2.2874 (avg 1.9556) | acc 28.5% (avg 28.7%) | lr 3.00e-04 | 10.7 step/s
  step   7900 | loss 2.3905 (avg 1.9480) | acc 29.4% (avg 28.7%) | lr 3.00e-04 | 10.7 step/s
  step   8000 | loss 1.8929 (avg 1.9299) | acc 29.3% (avg 28.7%) | lr 3.00e-04 | 10.7 step/s
  ╔══ sample (seed=8000) ══╗
  ║ Once upon a time there wos dperseeede  efteeee  ete e    e    e      e
  ║ collapse: 100%
  ╚══════════════════════════════╝
  step   8100 | loss 1.7265 (avg 1.9407) | acc 28.5% (avg 29.0%) | lr 3.00e-04 | 10.7 step/s
  step   8200 | loss 2.1090 (avg 1.9746) | acc 28.0% (avg 28.6%) | lr 3.00e-04 | 10.7 step/s
  step   8300 | loss 2.0476 (avg 1.9373) | acc 27.4% (avg 29.0%) | lr 3.00e-04 | 10.7 step/s
  step   8400 | loss 1.7010 (avg 1.9214) | acc 26.9% (avg 28.8%) | lr 3.00e-04 | 10.7 step/s
  step   8500 | loss 2.1919 (avg 1.9591) | acc 28.1% (avg 28.9%) | lr 3.00e-04 | 10.7 step/s
  step   8600 | loss 1.2647 (avg 1.9777) | acc 27.7% (avg 28.8%) | lr 3.00e-04 | 10.7 step/s
  step   8700 | loss 1.8608 (avg 1.9194) | acc 30.1% (avg 28.9%) | lr 3.00e-04 | 10.7 step/s
  step   8800 | loss 1.7973 (avg 1.8881) | acc 31.8% (avg 29.0%) | lr 3.00e-04 | 10.7 step/s
  step   8900 | loss 2.0637 (avg 1.9000) | acc 31.1% (avg 29.1%) | lr 3.00e-04 | 10.7 step/s
  step   9000 | loss 1.6089 (avg 1.8996) | acc 29.0% (avg 28.9%) | lr 3.00e-04 | 10.7 step/s
  step   9100 | loss 2.3126 (avg 1.9104) | acc 29.0% (avg 29.0%) | lr 3.00e-04 | 10.7 step/s
  step   9200 | loss 1.5791 (avg 1.9093) | acc 32.9% (avg 29.0%) | lr 3.00e-04 | 10.7 step/s
  step   9300 | loss 1.7902 (avg 1.8960) | acc 28.6% (avg 29.2%) | lr 3.00e-04 | 10.7 step/s
  step   9400 | loss 1.8613 (avg 1.9214) | acc 30.8% (avg 29.0%) | lr 3.00e-04 | 10.7 step/s
  step   9500 | loss 2.1701 (avg 1.9344) | acc 27.5% (avg 29.4%) | lr 3.00e-04 | 10.8 step/s
  step   9600 | loss 2.0838 (avg 1.8822) | acc 27.7% (avg 29.3%) | lr 3.00e-04 | 10.8 step/s
  step   9700 | loss 1.4937 (avg 1.8606) | acc 27.2% (avg 29.3%) | lr 3.00e-04 | 10.8 step/s
  step   9800 | loss 2.3377 (avg 1.8641) | acc 27.4% (avg 29.4%) | lr 3.00e-04 | 10.8 step/s
  step   9900 | loss 1.8963 (avg 1.9289) | acc 29.4% (avg 29.3%) | lr 3.00e-04 | 10.8 step/s
  step  10000 | loss 1.8828 (avg 1.8885) | acc 28.6% (avg 29.3%) | lr 3.00e-04 | 10.8 step/s
  ╔══ sample (seed=10000) ══╗
  ║ Once upon a time, there was a little      a e ee ee e   eeee e      e 
  ║ collapse: 100%
  ╚══════════════════════════════╝
  ✓ saved ./checkpoints/torus_english_step_10000.pt
  step  10100 | loss 1.7123 (avg 1.9235) | acc 27.0% (avg 29.3%) | lr 3.00e-04 | 10.8 step/s
  step  10200 | loss 1.6575 (avg 1.8486) | acc 31.2% (avg 29.8%) | lr 3.00e-04 | 10.8 step/s
  step  10300 | loss 1.9161 (avg 1.8857) | acc 29.0% (avg 29.6%) | lr 3.00e-04 | 10.8 step/s
  step  10400 | loss 2.0486 (avg 1.8231) | acc 30.5% (avg 29.9%) | lr 3.00e-04 | 10.8 step/s
  step  10500 | loss 1.7844 (avg 1.8982) | acc 31.4% (avg 29.7%) | lr 3.00e-04 | 10.8 step/s
  step  10600 | loss 1.9437 (avg 1.9044) | acc 30.2% (avg 30.2%) | lr 3.00e-04 | 10.8 step/s
  step  10700 | loss 2.3476 (avg 1.8957) | acc 27.9% (avg 29.9%) | lr 3.00e-04 | 10.8 step/s
  step  10800 | loss 2.2675 (avg 1.9016) | acc 31.4% (avg 30.4%) | lr 3.00e-04 | 10.8 step/s
  step  10900 | loss 2.2604 (avg 1.8519) | acc 31.0% (avg 30.4%) | lr 3.00e-04 | 10.8 step/s
  step  11000 | loss 1.9821 (avg 1.8925) | acc 28.8% (avg 30.3%) | lr 3.00e-04 | 10.8 step/s
  step  11100 | loss 2.1308 (avg 1.8461) | acc 30.5% (avg 30.7%) | lr 3.00e-04 | 10.8 step/s
  step  11200 | loss 1.9662 (avg 1.8182) | acc 30.1% (avg 30.5%) | lr 3.00e-04 | 10.8 step/s
  step  11300 | loss 1.4348 (avg 1.8872) | acc 30.5% (avg 30.7%) | lr 3.00e-04 | 10.8 step/s
  step  11400 | loss 1.7903 (avg 1.8524) | acc 31.6% (avg 30.9%) | lr 3.00e-04 | 10.8 step/s
  step  11500 | loss 1.4771 (avg 1.8775) | acc 28.2% (avg 31.0%) | lr 3.00e-04 | 10.8 step/s
  step  11600 | loss 2.2017 (avg 1.8349) | acc 31.9% (avg 31.2%) | lr 3.00e-04 | 10.8 step/s
  step  11700 | loss 1.6287 (avg 1.7898) | acc 34.6% (avg 31.5%) | lr 3.00e-04 | 10.8 step/s
  step  11800 | loss 1.4752 (avg 1.8267) | acc 32.2% (avg 31.1%) | lr 3.00e-04 | 10.8 step/s
  step  11900 | loss 1.3776 (avg 1.7885) | acc 29.6% (avg 31.9%) | lr 3.00e-04 | 10.8 step/s
  step  12000 | loss 1.2251 (avg 1.8313) | acc 31.3% (avg 31.6%) | lr 3.00e-04 | 10.8 step/s
  ╔══ sample (seed=12000) ══╗
  ║ Once upon a time, there was a m   tlede lan  aisnneee erig  eh ne h ui
  ║ collapse: 100%
  ╚══════════════════════════════╝
  step  12100 | loss 1.6403 (avg 1.7977) | acc 32.2% (avg 31.9%) | lr 3.00e-04 | 10.8 step/s
  step  12200 | loss 2.0586 (avg 1.8576) | acc 32.7% (avg 32.1%) | lr 3.00e-04 | 10.8 step/s
  step  12300 | loss 2.0410 (avg 1.8343) | acc 32.0% (avg 32.0%) | lr 3.00e-04 | 10.8 step/s
  step  12400 | loss 1.9938 (avg 1.8340) | acc 32.2% (avg 32.2%) | lr 3.00e-04 | 10.8 step/s
  step  12500 | loss 2.0740 (avg 1.7409) | acc 31.0% (avg 32.5%) | lr 3.00e-04 | 10.8 step/s
  step  12600 | loss 1.5402 (avg 1.7918) | acc 35.2% (avg 32.6%) | lr 3.00e-04 | 10.8 step/s
  step  12700 | loss 1.5163 (avg 1.7717) | acc 34.4% (avg 32.7%) | lr 3.00e-04 | 10.8 step/s
  step  12800 | loss 1.9560 (avg 1.7807) | acc 33.7% (avg 32.9%) | lr 3.00e-04 | 10.8 step/s
  step  12900 | loss 1.6902 (avg 1.8302) | acc 32.8% (avg 33.0%) | lr 3.00e-04 | 10.8 step/s
  step  13000 | loss 1.3398 (avg 1.7699) | acc 35.7% (avg 33.1%) | lr 3.00e-04 | 10.8 step/s
  step  13100 | loss 1.8984 (avg 1.7689) | acc 30.4% (avg 33.4%) | lr 3.00e-04 | 10.8 step/s
  step  13200 | loss 1.7517 (avg 1.7464) | acc 35.6% (avg 33.7%) | lr 3.00e-04 | 10.8 step/s
  step  13300 | loss 1.5744 (avg 1.7736) | acc 38.2% (avg 33.8%) | lr 3.00e-04 | 10.8 step/s
  step  13400 | loss 1.6670 (avg 1.7084) | acc 37.8% (avg 33.9%) | lr 3.00e-04 | 10.8 step/s
  step  13500 | loss 1.2638 (avg 1.7754) | acc 38.4% (avg 33.6%) | lr 3.00e-04 | 10.8 step/s
  step  13600 | loss 1.4833 (avg 1.7915) | acc 36.9% (avg 33.9%) | lr 3.00e-04 | 10.8 step/s
  step  13700 | loss 1.9286 (avg 1.7525) | acc 34.6% (avg 34.2%) | lr 3.00e-04 | 10.8 step/s
  step  13800 | loss 1.6560 (avg 1.7702) | acc 36.0% (avg 33.8%) | lr 3.00e-04 | 10.8 step/s
  step  13900 | loss 2.0094 (avg 1.7295) | acc 34.1% (avg 34.1%) | lr 3.00e-04 | 10.8 step/s
  step  14000 | loss 1.8165 (avg 1.6878) | acc 31.8% (avg 34.6%) | lr 3.00e-04 | 10.8 step/s
  ╔══ sample (seed=14000) ══╗
  ║ Once upon a time, there was a little litoy named TLmly.  ie loved od t
  ║ collapse: 100%
  ╚══════════════════════════════╝
  step  14100 | loss 1.9775 (avg 1.7491) | acc 33.2% (avg 34.9%) | lr 3.00e-04 | 10.8 step/s
  step  14200 | loss 1.6283 (avg 1.7922) | acc 30.3% (avg 34.7%) | lr 3.00e-04 | 10.8 step/s
  step  14300 | loss 1.8401 (avg 1.7273) | acc 34.1% (avg 34.9%) | lr 3.00e-04 | 10.8 step/s
  step  14400 | loss 1.5682 (avg 1.6831) | acc 38.1% (avg 35.8%) | lr 3.00e-04 | 10.8 step/s
  step  14500 | loss 2.2225 (avg 1.7393) | acc 34.2% (avg 35.6%) | lr 3.00e-04 | 10.8 step/s
  step  14600 | loss 1.7397 (avg 1.7038) | acc 36.7% (avg 36.1%) | lr 3.00e-04 | 10.8 step/s
  step  14700 | loss 1.2939 (avg 1.7056) | acc 39.8% (avg 36.2%) | lr 3.00e-04 | 10.8 step/s
  step  14800 | loss 1.2323 (avg 1.7157) | acc 39.5% (avg 35.9%) | lr 3.00e-04 | 10.8 step/s
  step  14900 | loss 1.4269 (avg 1.7148) | acc 39.8% (avg 36.5%) | lr 3.00e-04 | 10.8 step/s
  step  15000 | loss 1.3525 (avg 1.6848) | acc 39.5% (avg 36.5%) | lr 3.00e-04 | 10.8 step/s
  ✓ saved ./checkpoints/torus_english_step_15000.pt
  step  15100 | loss 1.6750 (avg 1.7033) | acc 34.0% (avg 36.3%) | lr 3.00e-04 | 10.8 step/s
  step  15200 | loss 2.5936 (avg 1.7092) | acc 31.3% (avg 36.4%) | lr 3.00e-04 | 10.8 step/s
  step  15300 | loss 1.3960 (avg 1.6441) | acc 37.8% (avg 36.8%) | lr 3.00e-04 | 10.8 step/s
  step  15400 | loss 2.2504 (avg 1.6554) | acc 34.0% (avg 37.1%) | lr 3.00e-04 | 10.8 step/s
  step  15500 | loss 1.9727 (avg 1.6582) | acc 35.8% (avg 36.8%) | lr 3.00e-04 | 10.8 step/s
  step  15600 | loss 1.6928 (avg 1.6172) | acc 37.3% (avg 37.6%) | lr 3.00e-04 | 10.8 step/s
  step  15700 | loss 1.8795 (avg 1.6773) | acc 33.3% (avg 37.0%) | lr 3.00e-04 | 10.8 step/s
  step  15800 | loss 1.5801 (avg 1.6369) | acc 37.1% (avg 37.7%) | lr 3.00e-04 | 10.8 step/s
  step  15900 | loss 1.6209 (avg 1.7080) | acc 42.0% (avg 36.9%) | lr 3.00e-04 | 10.8 step/s
  step  16000 | loss 1.9312 (avg 1.6906) | acc 38.2% (avg 37.3%) | lr 3.00e-04 | 10.8 step/s
  ╔══ sample (seed=16000) ══╗
  ║ Once there was a atleg ela leaiewded.  wito tond a  tto thom  ilke   e
  ║ collapse: 100%
  ╚══════════════════════════════╝
  step  16100 | loss 1.8185 (avg 1.6823) | acc 36.5% (avg 37.3%) | lr 3.00e-04 | 10.8 step/s
  step  16200 | loss 1.4914 (avg 1.6577) | acc 39.9% (avg 37.9%) | lr 3.00e-04 | 10.8 step/s
  step  16300 | loss 2.0395 (avg 1.6992) | acc 34.9% (avg 37.2%) | lr 3.00e-04 | 10.8 step/s
  step  16400 | loss 1.2916 (avg 1.6471) | acc 39.1% (avg 37.7%) | lr 3.00e-04 | 10.8 step/s
  step  16500 | loss 1.1964 (avg 1.6529) | acc 39.5% (avg 37.7%) | lr 3.00e-04 | 10.8 step/s
  step  16600 | loss 1.6079 (avg 1.6685) | acc 39.3% (avg 37.7%) | lr 3.00e-04 | 10.8 step/s
  step  16700 | loss 1.9314 (avg 1.6557) | acc 36.6% (avg 37.7%) | lr 3.00e-04 | 10.8 step/s
  step  16800 | loss 1.6105 (avg 1.6236) | acc 38.1% (avg 38.3%) | lr 3.00e-04 | 10.8 step/s
  step  16900 | loss 1.9546 (avg 1.5985) | acc 35.8% (avg 38.2%) | lr 3.00e-04 | 10.8 step/s
  step  17000 | loss 1.8159 (avg 1.6711) | acc 36.2% (avg 38.1%) | lr 3.00e-04 | 10.8 step/s
  step  17100 | loss 1.4803 (avg 1.6337) | acc 38.2% (avg 37.9%) | lr 3.00e-04 | 10.8 step/s
  step  17200 | loss 1.4658 (avg 1.6236) | acc 39.6% (avg 38.1%) | lr 3.00e-04 | 10.8 step/s
  step  17300 | loss 1.7238 (avg 1.6447) | acc 37.7% (avg 38.2%) | lr 3.00e-04 | 10.8 step/s
  step  17400 | loss 1.3916 (avg 1.6301) | acc 39.6% (avg 38.1%) | lr 3.00e-04 | 10.8 step/s
  step  17500 | loss 1.6556 (avg 1.5940) | acc 40.9% (avg 38.3%) | lr 3.00e-04 | 10.8 step/s
  step  17600 | loss 1.5779 (avg 1.6433) | acc 37.1% (avg 38.1%) | lr 3.00e-04 | 10.8 step/s
  step  17700 | loss 1.8649 (avg 1.6040) | acc 36.3% (avg 38.7%) | lr 3.00e-04 | 10.8 step/s
  step  17800 | loss 1.9882 (avg 1.6474) | acc 38.6% (avg 37.9%) | lr 3.00e-04 | 10.8 step/s
  step  17900 | loss 2.0596 (avg 1.5829) | acc 35.0% (avg 38.9%) | lr 3.00e-04 | 10.8 step/s
  step  18000 | loss 1.4505 (avg 1.6022) | acc 37.3% (avg 38.4%) | lr 3.00e-04 | 10.8 step/s
  ╔══ sample (seed=18000) ══╗
  ║ Once there was a biparent ot aa e  as thee hama  wind wat. ee s were t
  ║ collapse: 100%
  ╚══════════════════════════════╝
  step  18100 | loss 1.7041 (avg 1.6674) | acc 36.9% (avg 38.2%) | lr 3.00e-04 | 10.8 step/s
  step  18200 | loss 2.0726 (avg 1.6446) | acc 35.8% (avg 37.9%) | lr 3.00e-04 | 10.8 step/s
  step  18300 | loss 1.9683 (avg 1.6309) | acc 39.1% (avg 38.4%) | lr 3.00e-04 | 10.8 step/s
  step  18400 | loss 1.4723 (avg 1.6507) | acc 41.7% (avg 38.5%) | lr 3.00e-04 | 10.8 step/s
  step  18500 | loss 1.4441 (avg 1.6398) | acc 41.0% (avg 38.4%) | lr 3.00e-04 | 10.8 step/s
  step  18600 | loss 1.6788 (avg 1.6620) | acc 38.1% (avg 38.5%) | lr 3.00e-04 | 10.8 step/s
  step  18700 | loss 1.3676 (avg 1.6004) | acc 42.2% (avg 39.2%) | lr 3.00e-04 | 10.8 step/s
  step  18800 | loss 1.9520 (avg 1.6327) | acc 35.6% (avg 38.9%) | lr 3.00e-04 | 10.8 step/s
  step  18900 | loss 1.4832 (avg 1.6513) | acc 40.2% (avg 38.9%) | lr 3.00e-04 | 10.8 step/s
  step  19000 | loss 1.3025 (avg 1.6156) | acc 42.1% (avg 38.7%) | lr 3.00e-04 | 10.8 step/s
  step  19100 | loss 1.6024 (avg 1.6400) | acc 38.4% (avg 38.6%) | lr 3.00e-04 | 10.8 step/s
  step  19200 | loss 1.0827 (avg 1.6830) | acc 42.4% (avg 37.9%) | lr 3.00e-04 | 10.8 step/s
  step  19300 | loss 2.1404 (avg 1.6456) | acc 34.0% (avg 38.7%) | lr 3.00e-04 | 10.8 step/s
  step  19400 | loss 1.3291 (avg 1.5648) | acc 40.5% (avg 39.4%) | lr 3.00e-04 | 10.8 step/s
  step  19500 | loss 1.4549 (avg 1.6524) | acc 36.2% (avg 38.5%) | lr 3.00e-04 | 10.8 step/s
  step  19600 | loss 2.0491 (avg 1.6280) | acc 33.8% (avg 38.7%) | lr 3.00e-04 | 10.8 step/s
  step  19700 | loss 1.7683 (avg 1.5871) | acc 36.3% (avg 39.2%) | lr 3.00e-04 | 10.8 step/s
  step  19800 | loss 1.7780 (avg 1.5929) | acc 36.6% (avg 39.3%) | lr 3.00e-04 | 10.8 step/s
  step  19900 | loss 1.4404 (avg 1.5962) | acc 42.5% (avg 39.3%) | lr 3.00e-04 | 10.8 step/s
  step  20000 | loss 1.0817 (avg 1.6049) | acc 42.4% (avg 39.3%) | lr 3.00e-04 | 10.8 step/s
  ╔══ sample (seed=20000) ══╗
  ║ Once upon a time, there was a little P who He who Tside he nam e loved
  ║ collapse: 99%
  ╚══════════════════════════════╝
  ✓ saved ./checkpoints/torus_english_step_20000.pt
  step  20100 | loss 1.4875 (avg 1.6202) | acc 38.4% (avg 39.1%) | lr 3.00e-04 | 10.8 step/s
  step  20200 | loss 1.5726 (avg 1.6183) | acc 38.2% (avg 39.2%) | lr 3.00e-04 | 10.8 step/s
  step  20300 | loss 1.6984 (avg 1.5947) | acc 39.9% (avg 39.0%) | lr 3.00e-04 | 10.8 step/s
  step  20400 | loss 2.2216 (avg 1.6296) | acc 35.6% (avg 38.9%) | lr 3.00e-04 | 10.8 step/s
  step  20500 | loss 1.6760 (avg 1.6104) | acc 38.1% (avg 39.0%) | lr 3.00e-04 | 10.8 step/s
  step  20600 | loss 1.7945 (avg 1.6086) | acc 39.0% (avg 39.2%) | lr 3.00e-04 | 10.8 step/s
  step  20700 | loss 1.1007 (avg 1.5608) | acc 42.9% (avg 39.6%) | lr 3.00e-04 | 10.8 step/s
  step  20800 | loss 1.8637 (avg 1.6249) | acc 37.1% (avg 39.2%) | lr 3.00e-04 | 10.8 step/s
  step  20900 | loss 1.7785 (avg 1.6065) | acc 39.9% (avg 39.2%) | lr 3.00e-04 | 10.8 step/s
  step  21000 | loss 1.1988 (avg 1.6547) | acc 41.1% (avg 38.9%) | lr 3.00e-04 | 10.8 step/s
  step  21100 | loss 2.1681 (avg 1.6650) | acc 36.4% (avg 38.8%) | lr 3.00e-04 | 10.8 step/s
  step  21200 | loss 1.5778 (avg 1.6124) | acc 41.6% (avg 39.3%) | lr 3.00e-04 | 10.8 step/s
  step  21300 | loss 1.7365 (avg 1.6016) | acc 37.8% (avg 39.5%) | lr 3.00e-04 | 10.8 step/s
  step  21400 | loss 1.9102 (avg 1.5837) | acc 40.2% (avg 39.8%) | lr 3.00e-04 | 10.8 step/s
  step  21500 | loss 1.4857 (avg 1.5867) | acc 37.8% (avg 39.6%) | lr 3.00e-04 | 10.8 step/s
  step  21600 | loss 1.8682 (avg 1.6063) | acc 35.9% (avg 39.5%) | lr 3.00e-04 | 10.8 step/s
  step  21700 | loss 1.8674 (avg 1.5386) | acc 38.4% (avg 40.3%) | lr 3.00e-04 | 10.8 step/s
  step  21800 | loss 1.2639 (avg 1.5945) | acc 43.4% (avg 39.9%) | lr 3.00e-04 | 10.8 step/s
  step  21900 | loss 0.9426 (avg 1.5627) | acc 43.6% (avg 39.9%) | lr 3.00e-04 | 10.8 step/s
  step  22000 | loss 1.9798 (avg 1.5741) | acc 38.5% (avg 39.9%) | lr 3.00e-04 | 10.8 step/s
  ╔══ sample (seed=22000) ══╗
  ║ Once upon a time, there was a bi ie  who . Se w ay: e te  thad e are h
  ║ collapse: 100%
  ╚══════════════════════════════╝
  step  22100 | loss 1.3630 (avg 1.5803) | acc 45.5% (avg 40.0%) | lr 3.00e-04 | 10.8 step/s
  step  22200 | loss 1.4696 (avg 1.5874) | acc 40.2% (avg 39.7%) | lr 3.00e-04 | 10.8 step/s
  step  22300 | loss 1.2485 (avg 1.5828) | acc 40.6% (avg 39.8%) | lr 3.00e-04 | 10.8 step/s
  step  22400 | loss 1.1729 (avg 1.6059) | acc 40.2% (avg 39.7%) | lr 3.00e-04 | 10.8 step/s
  step  22500 | loss 1.1271 (avg 1.6134) | acc 45.9% (avg 39.3%) | lr 3.00e-04 | 10.8 step/s
  step  22600 | loss 1.6577 (avg 1.6000) | acc 39.7% (avg 39.5%) | lr 3.00e-04 | 10.8 step/s
  step  22700 | loss 1.0652 (avg 1.5595) | acc 47.1% (avg 40.1%) | lr 3.00e-04 | 10.8 step/s
  step  22800 | loss 1.2180 (avg 1.6027) | acc 44.5% (avg 39.9%) | lr 3.00e-04 | 10.8 step/s
  step  22900 | loss 1.5727 (avg 1.6386) | acc 41.4% (avg 39.4%) | lr 3.00e-04 | 10.8 step/s
  step  23000 | loss 1.8118 (avg 1.5688) | acc 40.4% (avg 40.2%) | lr 3.00e-04 | 10.8 step/s
  step  23100 | loss 1.3893 (avg 1.5451) | acc 46.0% (avg 40.4%) | lr 3.00e-04 | 10.8 step/s
  step  23200 | loss 1.6857 (avg 1.5734) | acc 38.6% (avg 40.5%) | lr 3.00e-04 | 10.8 step/s
  step  23300 | loss 1.4876 (avg 1.5903) | acc 41.0% (avg 39.9%) | lr 3.00e-04 | 10.8 step/s
  step  23400 | loss 1.4250 (avg 1.6133) | acc 42.1% (avg 39.7%) | lr 3.00e-04 | 10.8 step/s
  step  23500 | loss 1.8438 (avg 1.6096) | acc 36.8% (avg 39.9%) | lr 3.00e-04 | 10.8 step/s
  step  23600 | loss 1.2649 (avg 1.5528) | acc 38.2% (avg 40.3%) | lr 3.00e-04 | 10.8 step/s
  step  23700 | loss 2.0732 (avg 1.6098) | acc 36.2% (avg 40.2%) | lr 3.00e-04 | 10.8 step/s
  step  23800 | loss 1.1462 (avg 1.5208) | acc 40.9% (avg 40.8%) | lr 3.00e-04 | 10.8 step/s
  step  23900 | loss 1.6767 (avg 1.5688) | acc 37.6% (avg 40.5%) | lr 3.00e-04 | 10.8 step/s
  step  24000 | loss 1.4415 (avg 1.5053) | acc 41.0% (avg 40.7%) | lr 3.00e-04 | 10.8 step/s
  ╔══ sample (seed=24000) ══╗
  ║ Lily has  e o  a  aearon  to her a  a had  Loch Every tried to w food 
  ║ collapse: 100%
  ╚══════════════════════════════╝
  step  24100 | loss 1.1786 (avg 1.5795) | acc 43.9% (avg 40.3%) | lr 3.00e-04 | 10.8 step/s
  step  24200 | loss 1.6847 (avg 1.5920) | acc 41.5% (avg 40.1%) | lr 3.00e-04 | 10.8 step/s
  step  24300 | loss 1.7930 (avg 1.5250) | acc 39.8% (avg 40.8%) | lr 3.00e-04 | 10.8 step/s
  step  24400 | loss 0.9240 (avg 1.5357) | acc 53.1% (avg 40.6%) | lr 3.00e-04 | 10.8 step/s
  step  24500 | loss 1.9926 (avg 1.5487) | acc 38.0% (avg 40.2%) | lr 3.00e-04 | 10.8 step/s
  step  24600 | loss 1.6021 (avg 1.5658) | acc 40.1% (avg 40.8%) | lr 3.00e-04 | 10.8 step/s
  step  24700 | loss 1.4654 (avg 1.5823) | acc 40.9% (avg 40.3%) | lr 3.00e-04 | 10.8 step/s
  step  24800 | loss 1.6027 (avg 1.5110) | acc 40.6% (avg 40.9%) | lr 3.00e-04 | 10.8 step/s
  step  24900 | loss 1.3868 (avg 1.5529) | acc 40.4% (avg 40.6%) | lr 3.00e-04 | 10.8 step/s
  step  25000 | loss 1.5549 (avg 1.5389) | acc 37.7% (avg 40.9%) | lr 3.00e-04 | 10.8 step/s
  ✓ saved ./checkpoints/torus_english_step_25000.pt
  step  25100 | loss 1.9641 (avg 1.5843) | acc 35.1% (avg 40.4%) | lr 3.00e-04 | 10.8 step/s
  step  25200 | loss 2.1434 (avg 1.5834) | acc 33.8% (avg 40.4%) | lr 3.00e-04 | 10.8 step/s
  step  25300 | loss 1.8007 (avg 1.5672) | acc 35.4% (avg 40.8%) | lr 3.00e-04 | 10.8 step/s
  step  25400 | loss 1.3768 (avg 1.5635) | acc 46.9% (avg 40.9%) | lr 3.00e-04 | 10.8 step/s
  step  25500 | loss 2.1657 (avg 1.5983) | acc 38.0% (avg 40.7%) | lr 3.00e-04 | 10.8 step/s
  step  25600 | loss 1.7836 (avg 1.5283) | acc 33.7% (avg 41.3%) | lr 3.00e-04 | 10.8 step/s
  step  25700 | loss 1.3890 (avg 1.5752) | acc 41.2% (avg 40.7%) | lr 3.00e-04 | 10.8 step/s
  step  25800 | loss 1.3774 (avg 1.5894) | acc 41.5% (avg 40.3%) | lr 3.00e-04 | 10.8 step/s
  step  25900 | loss 1.2011 (avg 1.5319) | acc 37.4% (avg 40.8%) | lr 3.00e-04 | 10.8 step/s
  step  26000 | loss 1.7767 (avg 1.5251) | acc 41.3% (avg 41.0%) | lr 3.00e-04 | 10.8 step/s
  ╔══ sample (seed=26000) ══╗
  ║ Once upon a time, there was a little girl named Luly. She loved to pla
  ║ collapse: 100%
  ╚══════════════════════════════╝
  step  26100 | loss 0.9034 (avg 1.5412) | acc 50.7% (avg 41.0%) | lr 3.00e-04 | 10.8 step/s
  step  26200 | loss 1.6112 (avg 1.5264) | acc 42.3% (avg 41.3%) | lr 3.00e-04 | 10.8 step/s
  step  26300 | loss 1.3729 (avg 1.5484) | acc 45.4% (avg 40.9%) | lr 3.00e-04 | 10.8 step/s
  step  26400 | loss 1.8075 (avg 1.5581) | acc 37.4% (avg 41.1%) | lr 3.00e-04 | 10.8 step/s
  step  26500 | loss 1.7392 (avg 1.5211) | acc 37.1% (avg 41.1%) | lr 3.00e-04 | 10.8 step/s
  step  26600 | loss 1.2886 (avg 1.5597) | acc 42.5% (avg 41.0%) | lr 3.00e-04 | 10.8 step/s
  step  26700 | loss 1.3265 (avg 1.5202) | acc 45.9% (avg 41.4%) | lr 3.00e-04 | 10.8 step/s
  step  26800 | loss 0.8128 (avg 1.5339) | acc 51.2% (avg 41.5%) | lr 3.00e-04 | 10.8 step/s
  step  26900 | loss 1.4345 (avg 1.5631) | acc 41.1% (avg 41.3%) | lr 3.00e-04 | 10.8 step/s
  step  27000 | loss 1.7411 (avg 1.5501) | acc 40.1% (avg 41.1%) | lr 3.00e-04 | 10.8 step/s
  step  27100 | loss 1.6480 (avg 1.5111) | acc 40.9% (avg 41.8%) | lr 3.00e-04 | 10.8 step/s
  step  27200 | loss 2.0819 (avg 1.5756) | acc 35.9% (avg 41.3%) | lr 3.00e-04 | 10.8 step/s
  step  27300 | loss 1.4604 (avg 1.5638) | acc 42.6% (avg 41.3%) | lr 3.00e-04 | 10.8 step/s
  step  27400 | loss 2.0005 (avg 1.5114) | acc 38.1% (avg 41.7%) | lr 3.00e-04 | 10.8 step/s
  step  27500 | loss 1.4708 (avg 1.5369) | acc 43.0% (avg 41.5%) | lr 3.00e-04 | 10.8 step/s
  step  27600 | loss 1.7452 (avg 1.5511) | acc 40.6% (avg 41.5%) | lr 3.00e-04 | 10.8 step/s
  step  27700 | loss 0.8961 (avg 1.5359) | acc 47.2% (avg 41.4%) | lr 3.00e-04 | 10.8 step/s
  step  27800 | loss 1.4591 (avg 1.5033) | acc 46.8% (avg 41.9%) | lr 3.00e-04 | 10.8 step/s
  step  27900 | loss 1.7854 (avg 1.5255) | acc 35.5% (avg 41.9%) | lr 3.00e-04 | 10.8 step/s
  step  28000 | loss 1.7977 (avg 1.4993) | acc 35.6% (avg 42.2%) | lr 3.00e-04 | 10.8 step/s
  ╔══ sample (seed=28000) ══╗
  ║ Once upon a time, there was a little girl named Lil loved to saw he pe
  ║ collapse: 100%
  ╚══════════════════════════════╝
  step  28100 | loss 1.5240 (avg 1.5451) | acc 41.9% (avg 41.6%) | lr 3.00e-04 | 10.8 step/s
  step  28200 | loss 1.5210 (avg 1.4917) | acc 43.5% (avg 42.3%) | lr 3.00e-04 | 10.8 step/s
  step  28300 | loss 1.1865 (avg 1.4479) | acc 44.8% (avg 42.8%) | lr 3.00e-04 | 10.8 step/s
  step  28400 | loss 1.1102 (avg 1.5235) | acc 47.4% (avg 41.9%) | lr 3.00e-04 | 10.8 step/s
  step  28500 | loss 0.9865 (avg 1.5646) | acc 43.9% (avg 41.6%) | lr 3.00e-04 | 10.8 step/s
  step  28600 | loss 1.0393 (avg 1.5113) | acc 48.4% (avg 42.0%) | lr 3.00e-04 | 10.8 step/s
  step  28700 | loss 2.0672 (avg 1.5291) | acc 38.8% (avg 41.9%) | lr 3.00e-04 | 10.8 step/s
  step  28800 | loss 0.7531 (avg 1.5134) | acc 53.0% (avg 42.1%) | lr 3.00e-04 | 10.8 step/s
  step  28900 | loss 2.0014 (avg 1.5629) | acc 37.6% (avg 41.8%) | lr 3.00e-04 | 10.8 step/s
  step  29000 | loss 1.4114 (avg 1.5388) | acc 42.3% (avg 41.8%) | lr 3.00e-04 | 10.8 step/s
  step  29100 | loss 1.3093 (avg 1.5235) | acc 44.6% (avg 42.0%) | lr 3.00e-04 | 10.8 step/s
  step  29200 | loss 1.9126 (avg 1.5226) | acc 36.3% (avg 42.2%) | lr 3.00e-04 | 10.8 step/s
  step  29300 | loss 1.1449 (avg 1.4027) | acc 44.4% (avg 43.2%) | lr 3.00e-04 | 10.8 step/s
  step  29400 | loss 1.0508 (avg 1.5027) | acc 48.5% (avg 42.6%) | lr 3.00e-04 | 10.8 step/s
  step  29500 | loss 1.2730 (avg 1.4733) | acc 43.8% (avg 42.9%) | lr 3.00e-04 | 10.8 step/s
  step  29600 | loss 1.5985 (avg 1.5149) | acc 44.1% (avg 42.2%) | lr 3.00e-04 | 10.8 step/s
  step  29700 | loss 1.6654 (avg 1.4859) | acc 42.3% (avg 42.8%) | lr 3.00e-04 | 10.8 step/s
  step  29800 | loss 1.4497 (avg 1.5415) | acc 41.6% (avg 42.0%) | lr 3.00e-04 | 10.8 step/s
  step  29900 | loss 1.2646 (avg 1.5206) | acc 38.2% (avg 42.1%) | lr 3.00e-04 | 10.8 step/s
  step  30000 | loss 1.7830 (avg 1.4936) | acc 40.2% (avg 42.5%) | lr 3.00e-04 | 10.8 step/s
  ╔══ sample (seed=30000) ══╗
  ║ Once upon a time, there was a big zhe was  to wanted to tn a b it ey a
  ║ collapse: 100%
  ╚══════════════════════════════╝
  ✓ saved ./checkpoints/torus_english_step_30000.pt
  step  30100 | loss 1.6158 (avg 1.5034) | acc 43.0% (avg 42.8%) | lr 3.00e-04 | 10.8 step/s
  step  30200 | loss 1.8582 (avg 1.4789) | acc 38.7% (avg 42.6%) | lr 3.00e-04 | 10.8 step/s
  step  30300 | loss 1.1692 (avg 1.5043) | acc 46.7% (avg 42.4%) | lr 3.00e-04 | 10.8 step/s
  step  30400 | loss 1.0153 (avg 1.5426) | acc 45.7% (avg 41.7%) | lr 3.00e-04 | 10.8 step/s
  step  30500 | loss 1.9271 (avg 1.5234) | acc 36.9% (avg 42.4%) | lr 3.00e-04 | 10.8 step/s
  step  30600 | loss 1.6101 (avg 1.5010) | acc 43.2% (avg 42.9%) | lr 3.00e-04 | 10.8 step/s
  step  30700 | loss 1.0652 (avg 1.4899) | acc 49.7% (avg 42.6%) | lr 3.00e-04 | 10.8 step/s
  step  30800 | loss 1.8580 (avg 1.4524) | acc 39.6% (avg 43.2%) | lr 3.00e-04 | 10.8 step/s
  step  30900 | loss 1.7481 (avg 1.5313) | acc 40.0% (avg 42.6%) | lr 3.00e-04 | 10.8 step/s
