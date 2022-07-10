# 基于强化学习DQN算法的AirSim自动驾驶仿真

本readme文件用于描述哈尔滨工业大学（深圳）2022年春季学期《人工神经网络》课程，第20组（贺超、王德坤）课设简介及代码使用指南。


## 目录

- [课设简介](#课设简介)
- [AirSim操作流程](#AirSim操作流程)
- [开发环境安装](#开发环境安装)
	- [运行环境准备](#运行环境准备)
- [Badge](#badge)
- [Example Readmes](#example-readmes)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

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


## Badge

If your README is compliant with Standard-Readme and you're on GitHub, it would be great if you could add the badge. This allows people to link back to this Spec, and helps adoption of the README. The badge is **not required**.

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

To add in Markdown format, use this code:

```
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
```

## Example Readmes

To see how the specification has been applied, see the [example-readmes](example-readmes/).

## Related Efforts

- [Art of Readme](https://github.com/noffle/art-of-readme) - 💌 Learn the art of writing quality READMEs.
- [open-source-template](https://github.com/davidbgk/open-source-template/) - A README template to encourage open-source contributions.

## Maintainers

[@RichardLitt](https://github.com/RichardLitt).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/RichardLitt/standard-readme/issues/new) or submit PRs.

Standard Readme follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

### Contributors

This project exists thanks to all the people who contribute. 
<a href="https://github.com/RichardLitt/standard-readme/graphs/contributors"><img src="https://opencollective.com/standard-readme/contributors.svg?width=890&button=false" /></a>


## License

[MIT](LICENSE) © Richard Littauer
