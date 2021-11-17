# Try to play by yourself!
import gym
from pyglet.window import key
import numpy as np
import time
from gym.wrappers.monitoring.video_recorder import VideoRecorder


bool_do_not_quit = True  # Boolean to quit pyglet
scores = []  # Your gaming score
a = np.array([0.0, 0.0, 0.0])  # Actions


def env_desc(env):
    # Number of Dimensions in the Observable Space and number of Control Actions in the Environments
    print('Observation Space:', env.observation_space)
    print('Action Space:', env.action_space)
    print("Observation Space Param: 96x96x3 values for Red, Green and Blue pixels")
    print("Observation Space Highs:", np.mean(env.observation_space.high))
    print("Observation Space Lows:   ", np.mean(env.observation_space.low))


# Function that maps key presses to car movement
def key_press(k, mod):
    global bool_do_not_quit, a, restart
    if k == 0xff0d:
        restart = True
    if k == key.ESCAPE:
        bool_do_not_quit = False  # To Quit
    if k == key.Q:
        bool_do_not_quit = False  # To Quit
    if k == key.LEFT:
        a[0] = -1.0
    if k == key.RIGHT:
        a[0] = +1.0
    if k == key.UP:
        a[1] = +1.0
    if k == key.DOWN:
        a[2] = +0.8  # set 1.0 for wheels to block to zero rotation


# Function that maps key releases to car movement
def key_release(k, mod):
    global a
    if k == key.LEFT and a[0] == -1.0:
        a[0] = 0
    if k == key.RIGHT and a[0] == +1.0:
        a[0] = 0
    if k == key.UP:
        a[1] = 0
    if k == key.DOWN:
        a[2] = 0


def run_carRacing(policy=None, record_video=False):
    global bool_do_not_quit, a, restart
    env = gym.make('CarRacing-v0').env
    # env_desc(env)
    env.reset()
    env.render()
    video_recorder = None
    # Record Video
    if record_video:
        video_recorder = VideoRecorder(env, './video.mp4', enabled=True)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    while bool_do_not_quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        t1 = time.time()  # Trial timer
        while bool_do_not_quit:
            state, reward, done, info = env.step(a)
            # time.sleep(1/10)  # Slow down to 10fps for us poor little human!
            total_reward += reward
            if steps % 200 == 0 or done:
                print(f"Step: {steps} | Reward: {total_reward:+0.2f}, | Action: {a}")
            steps += 1
            # if not record_video:  # Faster, but you can as well call env.render() every time to play full window.
            #     env.render()
            env.render()
            if record_video:
                video_recorder.capture_frame()
            if done or restart:
                t1 = time.time() - t1
                scores.append(total_reward)
                scores.append(total_reward)
                print("Trial", len(scores), "| Score:", total_reward, '|', steps, "steps | %0.2fs." % t1)
                break
        if not bool_do_not_quit:
            scores.append(total_reward)
            print("Trial", len(scores), "| Score:", total_reward, '|', steps, "steps | %0.2fs." % t1)

    if record_video:
        video_recorder.close()
        video_recorder.enabled = False
    env.close()


run_carRacing(record_video=True)
