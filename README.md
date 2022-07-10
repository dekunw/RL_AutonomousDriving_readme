# 基于强化学习DQN算法的AirSim自动驾驶仿真

本readme文件用于描述哈尔滨工业大学（深圳）2022年春季学期《人工神经网络》课程，第20组（贺超、王德坤）课设简介及代码使用指南。


## 目录

- [课设简介](#课设简介)
- [AirSim操作流程](#AirSim操作流程)
- [开发环境安装](#开发环境安装)
	- [运行环境准备](#运行环境准备)
	- [回合设计实现和获取起始位置](#回合设计实现和获取起始位置)
	- [获取坐标信息](#获取坐标信息)
- [Badge](#badge)
- [开发者](#开发者)
- [参考文献](#参考文献)

## 课设简介

学习《人工神经网络》课程后，结合小组成员的兴趣，制作此课设，作为对该课程的总结，以自动驾驶的问题为例，学习和实践自动驾驶的强化深度学习的开发过程。本课设对强化学习、经验回放、目标网络和深度Q学习算法进行简单介绍，并利用上述工具在AirSim仿真环境下实现部分自动驾驶功能，并将简要介绍其代码实现过程。

## AirSim操作流程

Windows系统下双击AirSimNH.exe，提示选择要驾驶的设备，选择是为汽车，否为四旋翼无人机，选择是（汽车）即可。启动了AirSim界面。默认情况下，AirSim处于键盘控制模式。在键盘模式下，可以用方向键控制汽车的移动，控制窗口的观察场景，还可以显示或隐藏和前景有关的子视图。按退格键可以将汽车重置到起始位置。按功能键F1可以显示帮助。

## 开发环境安装

### 运行环境准备

安装Anaconda3中的jupyter，同时创建并激活代码运行环境gym，在环境gym中已安装库TensorFlow 2.0和gym。在jupyter中运行代码时，需要将环境切换到gym。在环境gym下，安装Python扩展库msgpack-rpc-python和airsim。其中msgpack-rpc-python是远程过程调用的库，airsim库使用这个库和上一节中的仿真窗口通信。

接下来，要让Python程序连接仿真窗口，并读取仿真环境的信息。返回的结果是一个airsim.CarState对象。可以进一步用car_state.speed读取汽车速度，用car_state.kenematis_estimated读取其动力学估计值。除此之外，AirSim还提供了一些API读取仿真环境中的信息。这样的方法一般被命名为airsim.Client.simXXX()，例如simGetImages()可以读取图像，simGetCollisionInfo()可以读取碰撞信息。

接下来介绍如何控制汽车的运行。首先，将AirSim从键盘模式切换到API控制模式，使得汽车的控制由Python API来接管。然后，在API模式下，可以通过airsim.CarClient类的setCarControls()方法控制汽车的运行。方法的参数是一个airsim.CarControls对象，其构造方法有以下参数：

throttle：float类型，表示油门

steering：float类型，表示方向盘转动。负数是逆时针转方向盘，正数是顺时针。

brake：float类型，表示刹车。

handbrake：bool类型，表示是否拉手刹。

### 回合设计实现和获取起始位置

有了奖励函数，自动驾驶问题就变成了强化学习问题。为了训练方便，我们进一步将自动驾驶问题建模为回合制的问题。回合的定义参考了奖励函数的定义。当出现下面任意一个状况时，回合结束：

1、汽车撞到其它东西；

2、汽车速度小于2；

3、汽车和路面中心的最小距离大于3.5时；

4、在设置回合最长时间的情况下，运行时间超过设置时间。

在训练过程中有必要限制回合的最长时间。基于同样的道理，在训练过程中最好选择不同的起点以避免陷入局部行为。同时，在回合开始应该先对汽车进行加速，使得其速度超过2，以免出现因为启动速度过小而是回合立即结束的情况。至此，我们已经初步设计了回合制的强化学习任务。

启动新回合的第一步是要在地图上随机选择一个起始点，这个逻辑有后面的函数get_start_pose()实现。设置起始位置时并没有设置速度，所以汽车可能会沿着设置前的速度继续行驶，甚至会翻车、碰撞。因此当brake_confirm为真，在设置位置前先预设置一次，此时刹车一段时间，然后再正式设置，这样可以让汽车在设置的位置速度为0。然后start_accelerate选项使得汽车从速度0开始加速，使得开始时汽车有一些速度，不至于一开始就判断成回合结束。最后还要记录回合的起始时间，以便后续判断回合是否结束。

然后来看一下如何确定起始位置并将汽车放在环境中任意的位置。代码清单5中函数get_start_pose()确定起始位置，当其参数random=True时，随机选择起始位置，不过要选在路上，并且车的起始朝向需要顺着路的方向。实现代码是先构造代表位置坐标的airsim.Vector3r对象，再用airsim.to_quaternion()函数确定汽车的朝向，用弧度值yaw表示。最后构造airsim.Pose对象，将其作为参数传给airsim.Client类的simSetVehiclePose()方法。

### 获取坐标信息

无论是在回合开始时在路上选择回合起始点，还是在回合过程中计算车到路的距离，都需要获取地图上路的坐标信息。get_roads()函数，根据街道地图返回各街道起始点坐标。函数get_roads()有个参数include_corners，当它为True时，返回的坐标包括在道路拐弯和交叉处的小斜线的坐标，这些小斜线会参与距离的计算；当它为False时，返回的坐标不包括那些小斜线的坐标，只包括较长的道路线段，这可用于回合起始位置确定时的道路选择。

至此，我们已经实现了AirSimCarEnv类，完成了“智能体/环境接口”中的环境接口部分。下图是AirSimNH的街道地图。

## Badge

If your README is compliant with Standard-Readme and you're on GitHub, it would be great if you could add the badge. This allows people to link back to this Spec, and helps adoption of the README. The badge is **not required**.

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

To add in Markdown format, use this code:

```
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
```


## 开发者

贺超      21S153124

王德坤    21B953022

## 参考文献

[1]Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. nature, 2015, 518(7540): 529-533.

[2]Mnih V, Kavukcuoglu K, Silver D, et al. Playing atari with deep reinforcement learning[J]. arXiv preprint arXiv:1312.5602, 2013.

[3]Sutton, R. & Barto, A. Reinforcement Learning: An Introduction (MIT Press, 1998)

[4]Thorndike, E. L. Animal Intelligence: Experimental studies (Macmillan, 1911)

[5]Schultz, W., Dayan, P. & Montague, P. R. A neural substrate of prediction and reward. Science 275, 1593–1599 (1997)

[6]Serre, T., Wolf, L. & Poggio, T. Object recognition with features inspired by visual cortex. Proc. IEEE. Comput. Soc. Conf. Comput. Vis. Pattern. Recognit. 994–1000 (2005)

[7]Fukushima, K. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. Biol. Cybern. 36, 193–202 (1980)

[8]Bengio, Y. Learning deep architectures for AI. Foundations and Trends in Machine Learning 2, 1–127 (2009)

[9]Krizhevsky, A., Sutskever, I. & Hinton, G. ImageNet classification with deep convolutional neural networks. Adv. Neural Inf. Process. Syst. 25, 1106–1114 (2012)

[10]Bellemare, M. G., Naddaf, Y., Veness, J. & Bowling, M. The arcade learning environment: An evaluation platform for general agents. J. Artif. Intell. Res. 47, 253–279 (2013)

[11]Bellemare, M. G., Veness, J. & Bowling, M. Investigating contingency awareness using Atari 2600 games. Proc. Conf. AAAI. Artif. Intell. 864–871 (2012)

[12]Bendor, D. & Wilson, M. A. Biasing the content of hippocampal replay during sleep. Nature Neurosci. 15, 1439–1444 (2012)

[13]Nair, V. & Hinton, G. E. Rectified linear units improve restricted Boltzmann machines. Proc. Int. Conf. Mach. Learn. 807–814 (2010)

