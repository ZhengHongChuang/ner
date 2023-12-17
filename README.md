# 基于Bert的中文命名实体识别

## 数据集
实验数据来自[CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020)。这是一个中文细粒度命名实体识别数据集，是基于清华大学开源的文本分类数据集THUCNEWS，选出部分数据进行细粒度标注得到的。该数据集的训练集、验证集和测试集的大小分别为10748，1343，1345，平均句子长度37.4字，最长50字。由于测试集不直接提供，考虑到leaderboard上提交次数有限，本项目使用CLUENER2020的验证集作为模型表现评判的测试集。

CLUENER2020共有10个不同的类别，包括：组织(organization)、人名(name)、地址(address)、公司(company)、政府(government)、书籍(book)、游戏(game)、电影(movie)、职位(position)和景点(scene)。

原始数据分别位于具体模型的/data/clue/路径下，train.json和test.json文件中，文件中的每一行是一条单独的数据，一条数据包括一个原始句子以及其上的标签，具体形式如下：
```JSON
{
	"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",
	"label": {
		"name": {
			"叶老桂": [
				[9, 11],
				[32, 34]
			]
		},
		"company": {
			"浙商银行": [
				[0, 3]
			]
		}
	}
}
```
## 模型

&bull; Betr-base-chinese+softmax
## 训练
!["每个epoch平均损失"](data:image/png;base64,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6NHq1KlTwP7cuHGjioqKwnJ/PvHEE0pNTdWkSZOOuVw478NBgwYpPT09YJ+Vl5frgw8+8O+z7OxslZaWau3atf5l3n77bXm9Xn/4dTpf+Nu0aZPeeust9erV67jrFBYWKiYmpkm3abj46quvdODAAf/fZSTsR8m0yo8ePVojRow47rJO2ofH+35ozf+f2dnZ+uSTTwKCvO/HzBlnnNExGxIJbB6EgnrPPPOM5Xa7rYULF1qff/65dfPNN1vJyckBo5zCxa233molJSVZK1assPbs2eOfDh06ZFmWZW3evNn67W9/a61Zs8batm2b9dJLL1knnXSSdeGFF9pceev9/Oc/t1asWGFt27bN+s9//mPl5ORYKSkp1t69ey3LsqxbbrnF6t+/v/X2229ba9assbKzs63s7Gybq247j8dj9e/f35o5c2bA/HDchxUVFdb69eut9evXW5KsOXPmWOvXr/ePgL3vvvus5ORk66WXXrI+/vhj66qrrrIGDRpkHT582P8al19+uTVq1Cjrgw8+sN577z3rlFNOsa699lq7NqmJY21jbW2tdeWVV1r9+vWzCgsLA/5t+kZOrlq1ynrooYeswsJCa8uWLdY///lPq3fv3tbUqVNt3rIGx9rGiooK6xe/+IVVUFBgbdu2zXrrrbess846yzrllFOs6upq/2s4eT8e7+/UsiyrrKzM6tKli/XII480Wd/p+/B43w+Wdfz/P+vq6qyhQ4da48ePtwoLC60lS5ZYvXv3tvLy8uzYpLBFAHSQP//5z1b//v2t+Ph4a+zYsdb7779vd0knRFKz0xNPPGFZlmUVFRVZF154odWzZ0/L7XZbJ598svXLX/7SKisrs7fwNpgyZYrVp08fKz4+3urbt681ZcoUa/Pmzf7nDx8+bP34xz+2evToYXXp0sX65je/ae3Zs8fGik/MG2+8YUmyNm7cGDA/HPfh8uXLm/27nDZtmmVZ5lQwd999t5WWlma53W7rsssua7LdBw4csK699lqrW7duVmJiojV9+nSroqLChq1p3rG2cdu2bS3+21y+fLllWZa1du1aKysry0pKSrISEhKs008/3br33nsDwpPdjrWNhw4dssaPH2/17t3b6tSpkzVgwADrpptuavJD2sn78Xh/p5ZlWX/961+tzp07W6WlpU3Wd/o+PN73g2W17v/P7du3WxMnTrQ6d+5spaSkWD//+c+tI0eOdPDWhDeXZVlWiBoXAQAA4EAcAwgAABBlCIAAAABRhgAIAAAQZQiAAAAAUYYACAAAEGUIgAAAAFGGAAgAABBlCIAAEERrdktjHpUqauyuBABaRgAEAACIMgRAAACAKBNndwEAEExeS3qyUHrxC+nAIal/knTDWVLOSaZ79pZXpLmXS39ZLRWVSaf2kn59oXRyz4bXWLZV+utaaWeZlNJFmjJU+q/hDc/XeqT5a6Q3NksHD0tp3aQfjJSuHtKwzIb90p8/kLZ+LZ2WIs26SBqY3EEfAgAcBwEQQER5Yr30+mYp73wpM0lav0eatVzqkdCwzJ/el35+rgl381ZLuW9IL0yR4mKkDfukvGXSzaOlcSdJH5dI970nJbmlyaeZ9WcvN/N/ca50Si9pd4VUWh1Yx8MfSj87R+rRWcp/V/rtO9KCqzrucwCAYyEAAogYtR7piULp4UnS8DQzr1+iVFgsvbBB+ubpZt5No6Vz+pn7v7lYuuJf0vJt0rjB0r8+kc7OkG48yzw/INm04v3jYxMAd5RKS7dK866Qsvo1vMfRfny2NDrD3J82UvrZEqmmTnLzvy4AB+C/IgARY2eZVF0nzXg1cP4Rr3Rar4bHvnAoSUkJJuRtKzWPt30tXTQwcP0R6dLTn0oer/TlASnW1RDuWnJKoy7llC7m9utqKb1bGzYIAEKEAAggYhyuM7dzL5dSuwY+1ylW+qq8/e/R2ha8uEZD7Fz1t16r/e8PAMHAKGAAEWNQshQfKxVXmuP/Gk+NW94+KWm4X15jBoMMSq5/jR7SR8WBr/tRsRlMEhtjBot4LWnt7lBvDQCEDi2AACJG13gzWndOgWRJGpkuVdaaYwC7xTeEwMfWma7fnp3NYI3kBOnigea5/xouTX1R+ts6Mwjkk73Ss59Jd55vns/oLn3jVDOo45fnma7ePZXS14fNMYQAEA4IgAAiyq1jzIjfJ9ZLuyqk7vHSkBRp+qiGLtjbxkp/XGWOGTy1l/TQBNNFLJll8y8zp4H52zpz/N4tYxpGAEsmDM770IwOLqs/rm/6qI7fVgA4US7LsjgqBUBU8J0HcPk0qbvb7moAwD4cAwgAABBlCIAAAABRhi5gAACAKEMLIAAAQJQhAAIAAEQZAiAAAECUIQACAABEGQIgAABAlCEAAgAARBkCIAAAQJQhAAIAAEQZAiAAAECU+X9YHt8bSd/EgAAAAABJRU5ErkJggg==)
## 项目结构

* ner
  * bert-base-chinese(预训练bert模型)
  * dataset(中文实体数据集)
  * model(模型)
  * weights(模型训练权重文件)
  * .gitattributes(lfs 大文件追踪目录)
  * test.py(测试代码)
  * train.py(训练代码)
## 第三方库
* pytorch
* transformers
* numpy
* tensorboard
* tqdm




