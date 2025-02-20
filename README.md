# Medicalrecommendation

整体架构为：   
![image](https://github.com/user-attachments/assets/d70bc60b-09e9-4e94-9fff-ea8fc445eb35)  

## 思路：
为了减少语言模型的幻觉，通过人为绑定药物避免推荐药物出错  

## 做法：
采用neo4j绑定药物及疾病等联系，通过数据库查询绑定prompt增强模型输出  
 
## 相关内容：
医药相关信息中为通过网络爬虫获取的药品用量禁忌等信息组成的表格

Neo4j数据库内容如图：可以通过疾病关键词查找到常用药物  
![image](https://github.com/user-attachments/assets/13f5184f-f015-4fae-9972-c46a993e2984)  

## 结果：
加入prompt后推荐药物能按照prompt显示：
![image](https://github.com/user-attachments/assets/85358178-c76b-4048-9000-3ce4bfa0b872)  

为了方便app调用，将接口穿透至公网，同时开放/chat、/medicineInfo相关接口，app运行效果如图
![6c54636908fc3fe3b8dcec1f1c9c5d3](https://github.com/user-attachments/assets/560c54bd-44f1-49d5-b460-12e436544f7c)





