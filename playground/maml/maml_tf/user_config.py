GCE_LOCALRUN_LOGROOT = '/tmp/' + "local/"
GCE_LOCAL_LOGROOT = '/tmp' + "/gce/"
# GCE_CODE_DIR = '/Users/master/bstadie_sandbox/gce_stuff/'
GCE_EXTRA_SETUP = """
export PYTHONPATH=/root/code/packages/:$PYTHONPATH
export PYTHONPATH=/root/code/e_MAML/:$PYTHONPATH;
pip install -e /root/code/rl-algs
pip install -e mock pygame moleskin waterbear params_proto 
"""
