## python基础

1. python是什么样的语言，和其他语言的区别。

    Python是一种解释型语言，边运行边编译。

    Python是一种动态语言。指在运行期间才去做数据类型检查的语言。不用给任何变量指定数据类型，该语言会在你第一次赋值给变量时，在内部将数据类型记录下来。 而相对于静态语言如C， Java之类的需要在程序中预先生命变量的数据类型，数据类型是在编译期间检查的。

    Python也是强类型定义语言：强制数据类型定义的语言。也就是说，一旦一个变量被指定了某个数据类型，如果不经过强制转换，那么它就永远是这个数据类型了。Java也是。

    Java语言虽然比较接近解释型语言的特征，但执行之前已经预先进行一次预编译，生成的代码是介于机器码和Java源代码之间的中介代码，运行的时候则由JVM(Java的虚拟机平台，可视为解释器)解释执行。它既保留源代码的高抽象、可移植的特点，又已经完成了对源代码的大部分预编译工作，所以执行起来比“纯解释型”程序要快许多。

2. python中的线程和进程，以及使用场景，你知道协程吗？

    编程语言中的线程和进程：


    (https://zhuanlan.zhihu.com/p/37175407)[https://zhuanlan.zhihu.com/p/37175407]
    ***
    系统的线程和进程：

    抽象回答： 进程是资源分配的最小单位，线程是CPU调度的最小单位。

    线程在进程下进行。一个进程可以包含多个线程。

    做个简单的比喻：进程=火车，线程=车厢
    - 线程在进程下行进（单纯的车厢无法运行）
    - 一个进程可以包含多个线程（一辆火车可以有多个车厢）
    - 不同进程间数据很难共享（一辆火车上的乘客很难换到另外一辆火车，比如站点换乘）
    - 同一进程下不同线程间数据很易共享（A车厢换到B车厢很容易）
    - 进程要比线程消耗更多的计算机资源（采用多列火车相比多个车厢更耗资源）
    - 进程间不会相互影响，一个线程挂掉将导致整个进程挂掉（一列火车不会影响到另外一列火车，但是如果一列火车上中间的一节车厢着火了，将影响到所有车厢）
    - 进程可以拓展到多机，进程最多适合多核（不同火车可以开在多个轨道上，同一火车的车厢不能在行进的不同的轨道上）
    - 进程使用的内存地址可以上锁，即一个线程使用某些共享内存时，其他线程必须等它结束，才能使用这一块内存。（比如火车上的洗手间）－**"互斥锁"**
    - 进程使用的内存地址可以限定使用量（比如火车上的餐厅，最多只允许多少人进入，如果满了需要在门口等，等有人出来了才能进去）－**“信号量”**

    > [https://www.zhihu.com/question/25532384](https://www.zhihu.com/question/25532384)

    **协程（Co-routine）**：

    简单点说协程是进程和线程的升级版,进程和线程都面临着内核态和用户态的切换问题而耗费许多切换时间,而协程就是用户自己控制切换（类似于系统中断）的时机,不再需要陷入系统的内核态.

    比如函数A，B没有互相调用，使用协程时，在执行A时可以随时中断去执行B，B也可以中断去执行A。类似于多线程。

    和多线程比，协程有何优势？最大的优势就是协程极高的执行效率。因为子程序切换不是线程切换，而是由程序自身控制，因此，没有线程切换的开销，和多线程比，**线程数量越多，协程的性能优势就越明显**。

    第二大优势就是**不需要多线程的锁机制**，因为只有一个线程，也不存在同时写变量冲突，在协程中控制共享资源不加锁，只需要判断状态就好了，所以执行效率比多线程高很多。

    多进程+协程，既充分利用多核，又充分发挥协程的高效率，可获得极高的性能。

    > [https://www.liaoxuefeng.com/wiki/897692888725344/923057403198272](https://www.liaoxuefeng.com/wiki/897692888725344/923057403198272)

    > [https://www.jianshu.com/p/7c851145ee4c](https://www.jianshu.com/p/7c851145ee4c)




3. GIL是什么，为什么会有GIL，去掉会怎样，有了GIL为什么还要给程序加锁？

    Global Interpreter Lock 线程全局解释器锁，为了保证线程安全而采取的独立线程运行的限制，既一个核只能载同一时间运行 一个线程。限制同一个进程中只有一个线程进入Python解释器（阻止线程并发). 

    由于GIL的存在，即便是多线程的程序也无法利用多核CPU的优势。不过I/O密集型程序依然适合使用多线程，因为大部分时间都在等待I/O，对CPU的依赖很低。

    名词解释：

    - CPU密集型 - 数学计算，搜索，图像处理等。
    - I/O密集型 - 文件操作，网络交互，数据库等。


    如果去掉（自述）：就会造成线程混乱，线程的输入对不上输出，就比如说同时用不同的输入运行一个深度学习模型，得到的输出不知道是属于哪个线程的，又或者根本得不到输出结果。

    如果去掉GIL： 可能会发生数据竞争,造成数据错乱，不能保证线程内执行代码的原子操作。

    去掉GIL使用大密度的锁来替代的话会造成单线程的性能下降。保留GIL既可以支持多线程，同时把单线程的性能优势最大的发挥出来。

    多线程中任务中，可能会发生多个线程同时对一个公共资源（如全局变量）进行操作的情况，这是就会发生混乱。为了避免这种情况，需要引入**线程锁**的概念。只有一个线程能处于上锁状态，当一个线程上锁之后，如果有另外一个线程试图获得锁，该线程就会挂起直到拥有锁的线程将锁释放。这样就保证了同时只有一个线程对公共资源进行访问或修改。

    有了GIL为什么还要给程序加锁？GIL只是解释器级别的锁，它能保证多个线程在修改同一个变量的时候不会发生崩溃，但是结果可能是错乱的。

    https://zhidao.baidu.com/question/717332562100066845.html

4. 迭代器、可迭代对象、生成器分别是什么？生成器的作用和使用场景？

    迭代器（iterator）：迭代是Python最强大的功能之一，是访问集合元素的一种方式。
    迭代器是一个可以记住遍历的位置的对象。
    

    迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。

    迭代器有两个基本的方法：`iter()` 和 `next()`。

    可迭代对象：

    内部含有`__iter__`方法的对象，就是可迭代对象

    生成器（generator）：

    在 Python 中，使用了 yield 的函数被称为生成器（generator）。跟普通函数不同的是，生成器是一个返回迭代器的函数，只能用于迭代操作，更简单点理解生成器就是一个迭代器。在调用生成器运行的过程中，每次遇到 yield 时函数会暂停并保存当前所有的运行信息，返回 yield 的值, 并在下一次执行 next() 方法时从当前位置继续运行。

    > [Python3 迭代器与生成器](https://www.runoob.com/python3/python3-iterator-generator.html)

    生成器（generator）作用于返回大量的但是只用一次的数据。

    生成器可以一定程度上实现协程。

5. python中的装饰器是什么？如何实现？使用场景？

    装饰器是可以在已知的对象上实现附加的功能的声明。 装饰器简单的实现就是有一个wrapper函数，这个函数的参数是一个需要装饰的对象。

    有了装饰器，我们就可以抽离出大量函数中与函数功能本身无关的雷同代码并继续重用。概括的讲，装饰器的作用就是为已经存在的对象添加额外的功能。

    如何实现：
    
    
    应用场景： 经常被用于有切面需求的场景，较为经典的有插入日志、性能测试、事务处理等。

6. python中的元类（metaclass）是什么？

    __metaclass__是创建类时起作用.


7. python中的可变对象和不可变对象之间的区别。

    可变对象是指，对象的内容是可变的。不可变的对象则相反，表示其内容不可变（变化的是变量引用的内存地址）。

    ~~Python里边的变量赋值是对某个内存地址的应用。~~

    strings, tuples, 和numbers是不可更改的对象，而 list, dict, set 等则是可以修改的对象。

    当一个引用传递给函数的时候,函数自动复制一份引用,这个函数里的引用和外边的引用没有半毛关系了.

    [https://www.runoob.com/note/46684](https://www.runoob.com/note/46684)

8. python的魔术方法你知道哪些，new，init，call的区别是什么?

    基本魔术方法：
    ```
    __new__(cls): 此方法是用来生成类的实例的，它是类里面第一个执行的函数。第一个参数是类，其他参数用来直接传递给init方法。new决定是否使用init。因为new可以调用其他类的构造方法或直接返回别的势力对象作为本类的实例。如果没有返回实例对象则init不会被调用。
    __init__ : 构造函数，在生成对象（实例被创建）时调用。
    __del__ : 析构函数，释放对象（实例被销毁）时使用。
    __str__ : 当实例化的类对象 被str() 函数调用时 自动触发此方法
    __repr__ : 打印，转换。定义当被repr() 调用时的行为
    __bytes__ : 定义当被bytes() 调用时的行为
    __bool__: 定义当被bool() 调用时的行为
    __hash__ ：定义当被hash() 调用时的行为
    __format__： 定义当被format() 调用时的行为

    __setitem__ : 按照索引赋值
    __getitem__: 按照索引获取值
    __len__: 获得长度。定义当被len() 调用时的行为
    __cmp__: 比较运算
    __call__: 函数调用。一个类实例也可以变成一个可调用对象，只需要实现一个特殊方法__call__()。作用是可以把实例对象用类似函数的形式表示，进一步模糊了函数和对象的概念。
    __add__: 加运算
    __sub__: 减运算
    __mul__: 乘运算
    __truediv__: 除运算
    __mod__: 求余运算
    __pow__: 乘方

    ```


9. django的中间件是什么（optional）

10. celery的原理，如何配置worker的权重

    通过中间商（broker）分发任务和消息到对应的worker，然后吧结果存在backend。Celery的通常用来做分布式应用的。
    如果broker使用的是rabbitMQ，和worker之间的通讯是通过amqp协议。

***



## 项目

1. 如何定位内存过高或CPU过高的问题

    - 使用`top` 命令返回cpu消耗top 10的进程。
    - 查看应用端口是否占过多内存/cpu消耗

2. 画一下你的项目的结构

3. 项目中你遇到的最大的困难是什么，是如何解决的？

4. 项目中你最有成就感的地方是什么？

    全栈开发。又是人力资源的缺失，做了很多从前段到后端，以及服务器维护等工作。

5. 你业余是怎么学习编程的？看哪些书？有自己做过什么项目？

    主要是看官方文档。学习这个东西主要是靠好奇心，我比较关注全栈方向，需要学习的领域还有很多，特别是在企业级开发中的技术应用。比如说怎么实现集群，分布式等，这都会驱使我去学习。

## web

1. HTTP GET/POST/PUT/PATCH之间的区别

    从语义上来说区别：
    - get 获取资源。返回实体主体。
    - post 发送数据（当然也能获取资源）
    - put 更新资源（已经存在的
    - patch 更新资源（如果不存在可以创建新的资源

    另外
    - head 类似get，但是响应中没有具体内容，用于获取报头（header）
    - delete 删除资源。
    - connect Http/1.1 协议中预留给能够将连接改为管道方式的代理服务器（！不是很懂）
    - options 允许客户端查看服务器性能。返回后端所支持的请求方法，也可以用来岑是服务器性能。
    - trace 回显服务器收到的请求，主要用于测试或诊断

    
    （附加）
    get/post的区别
    
    get，参数在url中，有可能不安全，但是便于用户在浏览器发送请求。请求参数有长度限制，不适合发送太大的数据

    post， 相对安全，请求参数放在请求体中，适合发送较多的数据。

2. 状态码的含义以及出现场景，301，302，404，500，502，504等

    - `1**` 信息，服务器收到请求，需要请求者继续执行操作
    - `2**` 成功
    - `3**` 重定向，需要进一步操作
        - `301` 永久移动，请求的资源被永久移动到新的URI，返回的信息包括新的URI，浏览器会重定向到新的URI，今后任何新的请求都会使用新的URI代替
        - `302` 临时移动，类似301，但是之后的请求依旧使用原有的URI
        - `304` 未修改。所请求的资源未修改，服务器返回304时不会返回任何资源，客户端会访问缓存。

    - `4**` 客户端错误
        - `400` bad request。用户请求语法错误，服务器无法理解。
        - `401` unauthorized。请求需要用户身份认证。
        - `403` Forbidden。服务器理解客户端的请求，但是拒绝执行此请求
        - `404` Not found
        - `405` method not allowed。 请求方法错误
        - `413` request entity too large。 请求体的大小过大。一般出现在上传文件的时候，单个上传请求中的文件过大，服务器拒绝。

    - `5**` 服务器错误
        - 500 Internal server error。服务器内部出错。一般是应用程序出bug了。
        - 501 Not implemented。 服务器不支持请求的功能，无法完成请求。
        - 502 Bad gateway。作为网关（cgi/fastcgi）或者代理工作的服务器（nginx）尝试执行请求时，从远程服务器接受到了一个无效的响应。在用nginx反向代理的时候，应用服务器没有运行时会出现这个错误。
        - 503 service unavailable
        - 504 gateway timeout。 网关或代理未及时从服务器获得返回数据。通常发生在应用服务器处理一个请求时花太多时间。这种耗时太长的任务最好是用queue做background job。




3. cookie和session的区别和联系

    Cookie    |   Session
    ----------|----------
    存储在客户端|存储在服务端
    不安全     |



4. 从url请求到返回，中间经历了什么

    1. 解析IP地址。（DNS等）
    2. 应用层发送HTTP请求
    3. 传输层， TCP协议传输报文，三次握手四次挥手
        - 3次握手 - 1： 发送方 ---SYN----> 接收方
        - 3次握手 - 2： 发送方 <--SYN/ACK- 接收方
        - 3次握手 - 3： 发送方 ---ACK----> 接收方
        - 连接成功

        四次挥手（TCP连接的释放）：
        - 客户端 发送 fin=1 seq=x。（今天就到这里吧）
        - 服务端获得fin=1，然后ack=x+1， seq=y发送回客户端，服务端进入close-wait状态。（做关闭准备）（好，等我结账好再走）
        - 服务端关闭好了之后，发送 fin=1 ack=x+1， seq=y。（好了，我们走吧）
        - 客户端收到，然后返回fin=1 ack=z+1，seq=h。（再见，下次约）
    4. 网络层IP协议查询mac地址，IP把TCP分割好的各种数据包发送给接收方。通过mac地址确认接收方。
    5. 服务器接受请求，再层层向上直到应用层，接受到HTTP请求后查找资源并返回报文。


5. HTTP和HTTPS的区别，HTTPS如何进行加密的

    HTTPS是在HTTP的基础上，在应用层和连阶层之间使用了SSL等对数据进行加密。HTTP使用80端口，HTTPS使用443端口。

    HTTPS通过非对称加密方法对数据进行加密。

## 数据库

1. mysql的索引是什么，如何建立索引，B+树的结构

2. mysql中的事务是什么，隔离等级是什么

3. 如何优化sql语句

4. mysql的性能优化等

## 操作系统

1. 堆和栈的区别

    从数据结构的角度解释：
    
    栈是一种后进先出的数据结构

    堆是一种经过排序的树形数据结构，每个节点都有一个值。通常我们所说的堆的数据结构是指二叉树。堆的特点是根节点的值最小（或最大），且根节点的两个树也是一个堆。由于堆的这个特性，常用来实现优先队列，堆的存取是随意的，

    从内存分配的角度解释：

    栈（stack）内存首先是一片内存区域。由系统自动分配。存储的都是局部变量。定义在函数内部的变量都是局部变量。所以在执行脚本的时候，函数先进栈，再定义变量。变量离开了作用域就会被释放。栈内存的更新速度较快，因为局部变量的生命周期比较短。栈的大小是由系统限制的，如果溢出会报错。

    堆（heap）由程序员自定义。存储的是数组和对象（数组其实也是对象）。堆的大小由开发者控制，会比栈更为灵活，但是产生的内存碎片可能也相对较多，而且如果没有适当的内存控制，会造成内存溢出。

2. 什么是io多路复用

    多路复用是指使用一个线程来检查多个文件描述符（Socket）的就绪状态，比如调用select和poll函数，传入多个文件描述符，如果有一个文件描述符就绪，则返回，否则阻塞直到超时。得到就绪状态后进行真正的操作可以在同一个线程里执行，也可以启动线程执行（比如使用线程池）。

    这样在处理1000个连接时，只需要1个线程监控就绪状态，对就绪的每个连接开一个线程处理就可以了，这样需要的线程数大大减少，减少了内存开销和上下文切换的CPU开销。

    常用的函数有`select`, `poll`, `epoll`
    基本上select有3个缺点:

    1. 连接数受限
    2. 查找配对速度慢
    3. 数据由内核拷贝到用户态

    poll改善了第一个缺点

    epoll改了三个缺点.

    参考知乎：https://www.zhihu.com/question/28594409



3. nginx的配置



## 算法

1. 找到整数列表的最大k个数，时间复杂度

2. 输入一维数组array和n，找出和值为n的任意两个元素（两数之和）

一遍哈希表。 n = x + y
tmp = dict()存储



3. 常见的排序算法，时间复杂度分析

4. 生成一个旋转矩阵

5. 二叉树，AVL树，红黑树

## 算法真题选

1. 已知一个A数组包含1～N，例如[1,2,3,4,5,...,100],B数组为从A中去除2个元素后并随机打乱顺序的长度为N-2的数组，快速求出这两个数字分别是什么。假设两个数字分别是x， y

Sum(A) - Sum(B) = x + y
然后就是两数之和的问题。

***
### 面试笔记

1. Flask和Django有什么区别？各有什么应用场景？

    Flask:
    - 小巧、灵活，让程序员自己决定定制哪些功能，非常适用于小型网站。
    - 使用Flask来开发大型网站也一样，开发的难度较大，代码架构需要自己设计，开发成本取决于开发者的能力和经验。
    - 应用于灵活定制组件，实现微服务，


    Django:
    - 大而全，功能极其强大，是Python web框架的先驱，用户多，第三方库极其丰富。
    - 非常适合企业级网站的开发，但是对于小型的微服务来说，总有“杀鸡焉有宰牛刀”的感觉，体量较大，非常臃肿，定制化程度没有Flask高，也没有Flask那么灵活。
    - Django提供一站式的服务，从模板，session，ORM，Auth等等都分配好了
    - 应用于开发大的应用系统（比如新闻类网站、商城、ERP等）

2. 什么是CI/CD？从trigger一个CI/CD会发生什么（CI/CD过程）？

    CI(Continous Integration),CD(Countinous Delivery) 分别指持续集成和持续交付。是一套实现软件的构建，测试，部署的自动化流程。

    1. 版本控制
    2. 持续集成：编译，验证，单元测试，集成测试等
    3. 持续交付：部署到QA/stage（测试）服务器，以及UAT
    4. 持续部署：部署到产品环境。

    

3. 用过什么前段框架？项目是做什么的？你在项目中是做什么的？

    主要用过Angular 2和react。
    1. 做过的Angular项目中，有一个是做了一年，项目是一个类似于LinkedIn的平台，用于PwC网络。用来管理项目的人力资源。
    2. React主要是一些小项目，一些专门做数据可视化的平台。部门里有一个团队做数据分析的，需要用到highchart之类的库来渲染数据。

4. 怎么做异步开发？

5. 什么是RESTful API？

    RESTful - Representational State Transfer

    一种互联网软件架构。它结构清晰、符合标准、易于理解、扩展方便，所以正得到越来越多网站的采用。

    设计指南：
    - URI只能包含名词
    - 等

6. 介绍一下Nginx和Gunicorn是用来做什么的？

    Nginx：
    - 静态HTTP服务器。可以将服务器上的静态文件（如HTML、图片）通过HTTP协议展现给客户端。
    - 反向代理服务器
    - 负载均衡。 基于反向代理，让nginx请求多个应用服务器，防止发生单个应用服务器崩溃而造成网站崩溃。
    - 虚拟主机。一个主机可以部署多个应用服务器在不同端口。

    https://www.cnblogs.com/shao-shuai/p/10131138.html

7. 什么是反向代理？

    客户端本来可以直接通过HTTP协议访问某网站应用服务器，网站管理员可以在中间加上一个Nginx，客户端请求Nginx，Nginx请求应用服务器，然后将结果返回给客户端，此时Nginx就是反向代理服务器。
    
8. MongoDB 有什么缺点？

9. 怎么设计测试用例？
    1. 边界条件分析
    2. 功能图
    3. 等价类划分
    4. 错误推测
    5. 因果图
    6. 场景法

10. 错误处理？