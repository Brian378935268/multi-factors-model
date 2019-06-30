# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:41:46 2018

@author: zhengyang
"""

from bs4 import BeautifulSoup

html = '''
<html>
    <title>hello world</title>
    <body>
        <li class='one' id='1'>the first line</li>
        <li class='two' id='2'>the second line</li>
        <li class='three' id='3'>the third line</li>
    </body>
</html>
'''

# 构建文档树
root = BeautifulSoup(html, 'lxml')

# 格式化输出
print(root.prettify())

# 获取某个标签
tag = root.title

# 标签的名称
name = tag.name

# 获取标签的对应的文本内容，这里是可以当普通字符串处理
text = tag.string
text.split(' ')

# 如果有多个重复标签，普通方法只会获取第一个
li_tag = root.li

# 如果想获取所有的标签
li_tags = root.find_all('li')

# 获取属性
li_tag['class']   # 获取特定属性
li_tag.attrs      # 获取所有属性

# 判断是否存在某属性
li_tag.has_attr('id')

# 尝试 .content 属性获取所有子节点
root.html.contents
root.body.contents

# 使用 .children 返回的不是列表，是一个生成器
root.body.children
for item in root.body.children:
    print(item)

# descendants 包含孙节点
for child in root.body.descendants:
    print(child)

# 遍历每个字符串内容
for string in root.strings:
    print(string)
    
# 如果想要把空格去掉
for string in root.stripped_strings :
    print(string)
    
# 获取单个父对象
root.title.string.parent

# 获取所有父对象
for p in root.title.string.parents:
    print(p.name)

# 获取后一个兄弟节点
root.body.li.next_sibling

# 获取后面所有的兄弟节点
for s in root.body.li.next_siblings:
    print(s)










