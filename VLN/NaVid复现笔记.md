## NaVID模型结构：

![image-20250302201018311](https://github.com/Molion121/llm_related121/blob/main/VLN/image-20250302165954686.png)

模型训练的权重为两个部分，一个是BERT另一个是Vicuna-7B，按照LLaMA-VID的Instruction Tuning范式

因此首先需要了解下 [LLaMA-VID](#section3)

### LLaMA-VID<a id="section3"></a>

正如它的名字，只需要两个token表示每一帧

首先通过视觉编码器抽取出相应的视觉embedding Xt

同时加上相应的用户指令作为I输入，借由相应的生成器生成对应的查询向量Qt=生成器（Xt，I），其中Qformer参照BLIP2

查询向量与相应的视觉进行交叉得到权重再乘以原先的向量就可以得到相应指令的结合图像表示，也就对应的代码这几步

![image-20250302204117161](C:\Users\92809\AppData\Roaming\Typora\typora-user-images\image-20250302204117161.png)

最终投影回到LLM语言空间

同时将视觉嵌入到各种token尺寸，最低为1

也就是说**总的token最低可以仅为2，也正对应文章题目**

与LLama-Vid不同，如果将每一帧只是作为2个token输入，这不适用与Navid

![image-20250302165954686](C:\Users\92809\AppData\Roaming\Typora\typora-user-images\image-20250302165954686.png)

从类似LLaMA-VID可以看到类似的代码：

### ctxembed

基本如上所述，

![image-20250302204357689](C:\Users\92809\AppData\Roaming\Typora\typora-user-images\image-20250302204357689.png)

### visembed

论文里面写的是对于当前frame让其转换为64个token，而针对历史帧，将其转换为4帧

![](C:\Users\92809\AppData\Roaming\Typora\typora-user-images\image-20250302202804556.png)

将历史及当前帧组合成的token通过grid pool也就相当于2d的平均池化，同时通过投影层投影到LLM language空间中

![image-20250302203449892](C:\Users\92809\AppData\Roaming\Typora\typora-user-images\image-20250302203449892.png)

最终组合的token是将ctx embed和visembed组合

但这种方式会使得当存在很多帧的时候，会导致token越来越大，影响推理速度，所以后续再[UniNaVID](#section4)中改进了

### UniNaVID<a id="section4"></a>

后续就是当前帧仍然为64token，但是中间的64帧如果有就是为4token，超过64帧前就是1token

![image-20250302205228195](C:\Users\92809\AppData\Roaming\Typora\typora-user-images\image-20250302205228195.png)

算法只针对临界状况，也就是说如果当前帧要变成中间的64帧，就通过类似前面的grid pool变成4token

同样的中间段如果要变成64帧前同样变成1token，需要注意的是为了避免长视频导致超过64帧的部分越来越大

考虑到可能图像像素信息存在大量冗余，当临界要变成1token时，会对最近的long帧测一下二者的cos，

如果确实很相似，则在原有基础上叠加信息即可，如果不相似再额外给token

![image-20250302205714603](C:\Users\92809\AppData\Roaming\Typora\typora-user-images\image-20250302205714603.png)

## 复现遇到的问题：

1、版本问题：

给定的python版本为3.8

但是habitat-lab版本为0.1.7，其中的rl要求的tensorflow版本为1.13.1，不符合要求

同时考虑到cuda版本等，最终选定tensorflow版本为2.7.0

torch版本为2.0.1

torchvision 0.15.2

其他对应的降级模块看报错对应修改即可

2、文件目录问题

相应的eval.sh修改为对应的路径

相应的config文件修改为对应的路径

正常需要的文件目录为

![image-20250302210202440](C:\Users\92809\AppData\Roaming\Typora\typora-user-images\image-20250302210202440.png)

我们可以将相应的navid权重文件里面的config.json中的*mm_vision_tower*属性值修改为对应的路径，默认会通过调用build vision tower函数来获取属性值

![image-20250303102157758](C:\Users\92809\AppData\Roaming\Typora\typora-user-images\image-20250303102157758.png)

3、实验结果

生成相应的log以及video

![image-20250302210305006](C:\Users\92809\AppData\Roaming\Typora\typora-user-images\image-20250302210305006.png)
