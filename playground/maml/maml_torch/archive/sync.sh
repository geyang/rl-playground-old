#!/usr/bin/env bash
while [ 1 ]; do
    echo sync
    rsync -anP ge@slab-krypton.uchicago.edu:/home/ge/projects/reinforcement_learning_learning_notes/meta_learning_project/MAML_supervised/figures/debug_graph.pdf /Users/ge/Dropbox/Notes/reinforcement_learning_learning_notes/meta_learning_project/MAML_supervised/figures
    sleep 1.0
done
