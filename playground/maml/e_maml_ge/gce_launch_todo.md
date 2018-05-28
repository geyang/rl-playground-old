-  ✅ Added gce run script
-  ✅ Need to add a GCE `user_config.py` file
-  ✅ Use Logger for some of the logging (need to show you the code for review)
-  ✅ Verified GoalCheetah goal consistency
-  ✅ fixed a render error that shows at the end of the KrazyGridEnv run
-  ✅ try to save images inside the same folder as logger (need help, got help, done)
-  ✅ Performance review the tf.assign operations 
    - it is about 7 ~ 50 ms each time, no need to optimize atm b/c sampling takes much longer.
-  ✅ [DONE]: Implemented in-tf-session save and reload.
    - if have time, implement the in-tf-session weight/variable save and reload.
    - could potentially combine with others to remove session overhead. Not significant

-  ✅ remove Moleskin
-  ✅ remove ledger
-  ✅ remove Dashboard
-  ✅ get rcall to work
-  ✅ fix local gsutils (move to ~/google-cloud-sdk folder)
- [ ] Now try to fix the testing, to report on a fixed set of tasks
