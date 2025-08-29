# sim2sim

This repository contains the files I have written 
when attempting to create a *sim2sim* demo.

This demo involves transferring 
[ManiSkill](https://github.com/haosulab/ManiSkill)'s
`PickCube-v1` to 
[MuJoCo](https://github.com/google-deepmind/mujoco).
Doing this can be broken down into several steps:

1. Train a policy for `PickCube-v1` in MuJoCo,
    use the 
    [official PPO baseline script](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo) 
    with the following command line:

    ```
    python ppo_rgb.py --env_id="PickCube-v1" \
        --num_envs=256 --update_epochs=8 --num_minibatches=8 \
        --total_timesteps=10_000_000
    ```

2. Convert the yielding `final_ckpt.pt` to ONNX format,
    the `export_ppo_rgb_to_onnx.py` is created for this
    purpose.

    ONNX (Open Neural Network Exchange) is a portable
    file format for storing computational graphs for
    neural network models. It can be visualized by
    [Netron](https://netron.app/), especially for small
    ones.

3. Build an MJCF scene equivalent to `PickCube-v1` in
    ManiSkill, this involves converting the `.glb` model
    for the ManiSkill table to MJCF format, which consists
    of two steps:

    - Convert the `.glb` model to `.obj` format with
        Blender, note that `.obj` may not fully support features
        in `.glb` (for example, Physically Based Rendering), so
        it might be necessary to manually handle the textures (a
        process called *texture baking*).
    - Use the [obj2mjcf](https://github.com/kevinzakka/obj2mjcf)
        script to convert the `.obj` file to MJCF format

    The MJCF scene can be seen in `run_onnx_policy.py`.

4. Write a Python script that runs the ONNX policy in MuJoCo.
    The script is `run_onnx_policy.py`.

## Domain Randomization

1. Camera randomization (camera pose)
2. Joint randomization (stiffness, friction... of the joints)
3. Lighting randomization (lighting)

## Curriculum Learning

As training a policy capable of domain randomization from scratch is too difficult, I adopted a curriculum-learning approach.


