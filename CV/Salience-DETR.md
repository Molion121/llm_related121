# Salience_DETR

代码部分

salience_detr forward函数：

## 1、初始化image target并进行预处理preprocess

先获取初始化image和target，针对此进行预处理preprocess

![image-20241010144912985](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241010144912985.png)

预处理preprocess（继承于DNDETRDetector -》DETRDetector）

mask将不是image size里面的tensor其他非本图像部分设置为1，boxes中设置为w，h的相关比例系数

![image-20241010154150306](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241010154150306.png)

通过self.backbone中的res50提取特征，然后经过self.neck也就对应的ChannelMapper类将维度都转换为256维度

position embedding就是通过PositionEmbeddingSine类

![image-20241010154700021](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241010154700021.png)

![image-20241010154604277](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241010154604277.png)

## 2、生成denoising

获取真实的label和box组成list，再生成noised result

![image-20241010154937177](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241010154937177.png)

每张图片里面的类别不一定相等，所以找到拥有最大类别的，再将最大类别数乘以要生成的noising_num，能得到相应的noising group数，而noising分为正负样本，所以也将真实的label和box需要重复两倍的denoising_groups数，

![image-20241010160651123](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241010160651123.png)

加入label噪声是根据对应的label阈值然后通过rand判断如果小于阈值则使用noised label，其中noised label是通过随机生成的其他类别，不小于阈值则会返回真实的label，所以noised labels中含有真假标签

加入box噪声是根据对应的box数量及对应的denoising group生成positive id，positive id和negative id间隔是box的数量，negative部分相比positive变化幅度会更大，也就是放大更多或者缩小更多，xyxy_boxes限制在0到1之间是因为前面的gt_box已经将其除以图像的wh，gt_box是相对于图像宽高的系数，大于1已经超出的图像范围

因为不同batch的gt数量不同，所以最终针对有效的部分会分配noised label and boxes，对于其他部分则默认为0，总数是对于每个batch中最大的gt数量乘以denoising group数量再乘以2，也就是对应的200，

![image-20241010164235594](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241010164235594.png)

![image-20241010204538042](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241010204538042.png)

### attention_mask:

![image-20241010202746548](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241010202746548.png)



## 3、transformer

valid_ratios 有效比例，即对应每个bs中，其在不同特征layer中有效的部分除以整体的图像宽高所得到的比例系数，均小于等于1，

![image-20241010210526475](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241010210526475.png)

### gen_encoder_output_proposals



对生成的候选框进行有效性检查，确保其坐标在 `[0.01, 0.99]` 之间（防止边界框过小或过大）：

对 `output_proposals` 进行反 sigmoid 变换：

output proposal使用掩码将无效的区域设置为无穷大（`inf`），防止这些区域影响后续的学习：无效区域指的是memory中的mask部分以及output proposal的非归一化部分

output_memory将编码器的 `memory`（特征图）乘以掩码，同时将proposal有效区域保留，忽略填充区域，忽略无效的proposal，是不是也就意味着忽略了背景模块

![image-20241010215145199](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241010215145199.png)

计算不是图像mask部分的token数量即valid token nums，再乘以层级的过滤系数得到focus的token总数，bs上可能存在不同，需要取最大值，同时针对不同的image level测算出token 总数

![image-20241011160813836](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241011160813836.png)



### encoder

encoder中在encoder layer中将选择出前300个query并进行attention，q k是选择出来的query+pos，而v是query，这里attention出来的query含有其他query的综合信息，将其替换到原先的query处，再与整体的图像进行attention，

encoder中将选择的token进行综合，是否考虑了位置因素？个体相对于全部的分数，个体相对选择出的分数，选择出的分数相对于整体的分数，相乘操作



memory 经过backbone+neck

![image-20241026135248482](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241026135248482.png)

memory经过transformer neck：

![image-20241026135644892](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20241026135644892.png)

### decoder

输入：

query-target加上了带有噪声的label query

reference_points加上了噪声的box query sigmoid，同时是encoder outputs coord

value-memory经过neck的feature

attn mask将不同的注意力遮起来

# Conditional DETR

关于backbone：只使用res5最后一层特征

![image-20240828163300414](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20240828163300414.png)

然后将这个特征转换为transformer所需维度 2048维度降维到256维度

![image-20240828163413736](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20240828163413736.png)

掩码部分：最初设置为全1，通过代码将真实图像区域外的部分，即不属于图像内容部分设置为0，因为图像的大小存在不同，需要填充至相同大小，在将不属于原有图像的部分值设置为0，忽略填充区域影响而关注图像真实区域

![image-20240829165509059](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20240829165509059.png)

positionembeddingsine部分：

通过掩码得到有效位置的标识，沿着对应轴得到对应位置嵌入计算正弦等

![image-20240829165736351](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20240829165736351.png)

query embed，将生成对应的num query数量也就是300，其中的维度与transformer中维度一致，

- **`num_queries`**：表示要生成的查询向量的数量，也就是嵌入字典的大小。每个查询将有一个对应的嵌入向量。
- **`embed_dim`**：表示每个查询向量的维度。每个查询将被表示为一个 `embed_dim` 维的向量。

在像DETR（Detection Transformer）这样的模型中，`query_embed` 通常用来表示一组“查询”向量，这些向量被输入到Transformer模型的解码器部分。每个查询向量可以被视为模型用于检测某个物体或理解某个特征的“提问”或“探针”。

在目标检测中，查询向量帮助模型在特征图中寻找感兴趣的区域或对象。这些嵌入向量通常没有固定的初始含义，而是通过训练，模型学习如何使用这些查询来从图像特征中提取有意义的信息。

nn.Embedding(num_queries, embed_dim) 通过为 num_queries 个查询生成嵌入向量，使模型能够在输入数据中定位和识别目标。

分类头和box头：

![image-20240829172743448](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20240829172743448.png)

分类头class embed：初始时使模型倾向于预测类别概率很低，使其倾向于背景类别。每个类别的概率为百分之一

![image-20240829172227845](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20240829172227845.png)

![image-20240829172839641](https://github.com/Molion121/llm_related121/blob/main/CV/img/image-20240829172839641.png)

