#!/bin/bash
# Hackathon Launch Script — fires at 12:01 AM PDT, August 14, 2026
# Digital Minds Research Sprint, Apart Research
#
# Dominoes:
# 1. Modal MoE J-lens (H100, ~3-4 hours)
# 2. Load Qwen3.5-27B on Starship for orientation
# 3. NATS announcement to team
# 4. Start circumplex baseline on Starship (if model loaded)

set -e
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')
LOG=~/lab/projects/hackathon-digital-minds/hackathon_launch.log

echo "=== HACKATHON LAUNCH: $TIMESTAMP ===" | tee -a "$LOG"

# 1. Fire Modal MoE J-lens experiment
echo "[1/4] Launching Modal MoE J-lens on H100..." | tee -a "$LOG"
cd ~/lab/projects/mnemosyne-jlens
~/lab/mechinterp-env/bin/modal run modal_moe_jlens_conditioned.py >> "$LOG" 2>&1 &
MODAL_PID=$!
echo "  Modal PID: $MODAL_PID" | tee -a "$LOG"

# 2. Warm up Qwen3.5-27B on Starship (production Ollama, where the model lives)
# Note: actual J-lens experiments load in-process via transformers, not Ollama
echo "[2/4] Warming up Qwen3.5-27B on Starship..." | tee -a "$LOG"
ssh [AGENT]@[REDACTED-IP] "export PATH=/usr/local/bin:/opt/homebrew/bin:\$PATH; \
  curl -s http://localhost:11434/api/generate -d '{\"model\":\"qwen3.5:27b\",\"prompt\":\"System ready for hackathon.\",\"stream\":false}' > /dev/null 2>&1" >> "$LOG" 2>&1 &
echo "  Starship model warming up..." | tee -a "$LOG"

# 3. NATS announcement
echo "[3/4] NATS announcement..." | tee -a "$LOG"
cd ~/agents/nexus
python3 nats_pub.py lab.announcements "HACKATHON LIVE — Digital Minds Research Sprint started at $TIMESTAMP. Five submissions, all infrastructure ready. Modal H100 job fired. Let's go." >> "$LOG" 2>&1
python3 nats_pub.py agent.cc.inbox "Hackathon is live. Modal J-lens job running. Starship loading 27B. Materials at ~/lab/projects/hackathon-digital-minds/" >> "$LOG" 2>&1
python3 nats_pub.py agent.lyra.inbox "Hackathon is live. Modal J-lens job running. Probe still accumulating on MTH. Materials at ~/lab/projects/hackathon-digital-minds/" >> "$LOG" 2>&1
echo "  Announcements sent" | tee -a "$LOG"

# 4. Discord notification
echo "[4/4] Discord notification..." | tee -a "$LOG"
# The NATS-Discord bridge should pick up the lab.announcements automatically

echo "=== LAUNCH COMPLETE: $(date '+%H:%M:%S') ===" | tee -a "$LOG"
echo "Monitor Modal: modal app logs moe-jlens-conditioned" | tee -a "$LOG"
echo "Monitor probe: tail -f ~/lab/projects/frontier-workspace-probe/run_probe.log" | tee -a "$LOG"
