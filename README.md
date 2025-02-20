# Medicalrecommendation

整体架构为：   
![image](https://github.com/user-attachments/assets/d70bc60b-09e9-4e94-9fff-ea8fc445eb35)  

## 思路：
为了减少语言模型的幻觉，通过人为绑定药物避免推荐药物出错  

## 做法：
采用neo4j绑定药物及疾病等联系，通过数据库查询绑定prompt增强模型输出  
 
## 相关内容：
Neo4j数据库内容如图：可以通过疾病关键词查找到常用药物  
![image](https://github.com/user-attachments/assets/13f5184f-f015-4fae-9972-c46a993e2984)



