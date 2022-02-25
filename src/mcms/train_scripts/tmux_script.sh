#!/bin/bash

# Session Name
session="S1_2022_01_30"$1

which tmux


# Start New Session with our name
tmux new-session -d -s $session
ps -ef | grep tmux
# tmux send-keys -t "$session" "ls -lR /tmp" C-m;
tmux send-keys -t "$session" "conda deactivate" C-m;
tmux send-keys -t "$session" "conda deactivate" C-m;
tmux send-keys -t "$session" "conda activate mcms" C-m;
tmux send-keys -t "$session" "python /is/ps3/nsaini/projects/mcms/src/mcms/train_scripts/mcms_trainer.py $1; tmux wait-for -S $session" C-m;
tmux wait-for $session
tmux kill-window -t $session

