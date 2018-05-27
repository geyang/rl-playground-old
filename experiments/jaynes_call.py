def rcall(_verbose=False,
          _s3_prefix="s3://ge-bair/jaynes-debug/",
          _code_name=None,
          _code_root="../",
          _file_mask=None,
          _excludes=None
          ):
    """
    rcall, returns a jaynes runner object already having the current codebase uploaded to s3.

    :param _code_name: The name for the tar ball. Be very careful when using a fixed name, b/c the ec2 launch is async.
    :param _s3_prefix: the s3 prefix starting with the protocol s3://bucket-name/prefix-1/prefix-2
    :param _code_root: default ../
    :type _excludes: full string patter for excluding directories
    :param _file_mask: default "./setup.py ./causal_infogan ./scripts"
    :param _verbose:
    :return:
    """
    from jaynes import Jaynes, templates

    code_mount = templates.S3Mount(
        name=_code_name, local=_code_root, s3_prefix=_s3_prefix, pypath=True, file_mask=_file_mask, excludes=_excludes,
        compress=True)

    J = Jaynes(
        launch_log="jaynes_launch.log",
        mounts=[code_mount],
    )
    # note: you can stop here to avoid re-upload the scripts
    J.run_local_setup(verbose=_verbose)

    def run(fn, *args,
            _mode='local',
            _ip=None,
            _as_daemon=False,
            _spot_price=0.97,
            _instance_prefix="jaynes-debug",
            _log_dir=None,
            _startup_script=tuple(),
            _sync_s3=False,
            _region="us-west-2",
            _instance_type="t2.micro",
            _use_gpu=False,
            _docker_image="python:3.6",
            _upload_interval=300,
            _pem_file="~/.ec2/ge-berkeley.pem",
            **kwargs, ):
        """
        :param fn: the main function that is called in the docker
        :param args: arguments
        :param kwargs: keyword arguments for the function
        :param _mode:
        :param _ip: the ip address of the instance when used in ssh mode
        :param _as_daemon: if True, use popen to run the ssh so that it does not hang
        :param _spot_price: The spot price for the instances. None -> uses reserved instances instead
        :param _region:
        :param _instance_type:
        :param _instance_prefix: the tag.Name for the ec2 instance.
        :param _sync_s3: default False, whether to sync s3 locally.
        :param _use_gpu: bool, whether or not to use GPU. default to True
        :param _log_dir: the log directory that the python script sees
        :param _startup_script:
        :param _pem_file: the path to the pem file for ssh connections
        :param _docker_image: the docker image to use. default to "python:3.6"
        :param _upload_interval: the interval for the s3 upload.
        :return:
        """
        from copy import deepcopy
        if not _use_gpu:
            print('WARNING: NOT using GPU.')
        if _mode == "ssh":
            assert _ip is not None, "need _ip address under ssh mode"

        output_mount = templates.S3UploadMount(
            docker_abs=_log_dir, s3_prefix=_s3_prefix, local=None, interval=_upload_interval, sync_s3=_sync_s3)

        j_ = deepcopy(J)
        j_.mount(output_mount)

        j_.set_docker(
            docker=templates.DockerRun(_docker_image,
                                       name=_instance_prefix.replace('/', ".")[:30],
                                       pypath=":".join(
                                           [m.pypath for m in j_.mounts if hasattr(m, "pypath") and m.pypath]),
                                       docker_startup_scripts=_startup_script,
                                       docker_mount=" ".join([m.docker_mount for m in j_.mounts]),
                                       use_gpu=_use_gpu).run(fn, *args, **kwargs)
        )
        if _mode is 'local':
            j_.launch_local_docker(verbose=True, delay=300)
        elif _mode is 'ssh':
            j_.make_launch_script(log_dir=output_mount.remote_abs, instance_tag=_instance_prefix + '@ssh', sudo=True,
                                  terminate_after_finish=False, region="us-west-2")
            if _verbose:
                print(j_.launch_script)
            j_.launch_ssh(ip_address=_ip, pem=_pem_file, script_dir=output_mount.remote_abs, verbose=_verbose,
                          detached=_as_daemon)
        elif _mode is 'spot':
            j_.make_launch_script(log_dir=output_mount.remote_abs, instance_tag=_instance_prefix, sudo=False,
                                  terminate_after_finish=True, region="us-west-2")
            if _verbose:
                print(j_.launch_script)
            j_.launch_ec2(region="us-west-2", image_id="ami-bd4fd7c5", instance_type=_instance_type,
                          key_name="ge-berkeley",
                          security_group="torch-gym-prebuilt", spot_price=_spot_price, verbose=_verbose,
                          iam_instance_profile_arn="arn:aws:iam::055406702465:instance-profile/main", dry=False)

    J.run = run
    return J


if __name__ == "__main__":
    pass
    # rcall(train, log_dir="hey", _mode="spot", _verbose=False)
