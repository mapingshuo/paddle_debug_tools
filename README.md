
# requirement

```
paddlepaddle
```

# install

```shell
python setup.py install
``` 

# usage

在paddle组网和执行优化算法后，在run program之前，添加如下两行代码

```python
from paddle_debug_tools import memory_tool
tool = memory_tool.MemoryEstimate(fluid.default_main_program(), batch_size=32)
tool.cal_memory()
```

之后在相应文件夹下会生成一个memory_anal.png的文件，类似如下：

![example memory usage](image/memory_anal.png?raw=true "example memory usage")

上图是example/test_memory_tool.py的执行结果

如果想要在浏览器上访问生成的图片，使用：

```python
from paddle_debug_tools import memory_tool
tool = memory_tool.MemoryEstimate(fluid.default_main_program(), batch_size=32)
tool.cal_memory(serve=True, port=8111)
```

如果想要将多个program的结果显示在同一张图片上：

```python
from paddle_debug_tools import memory_tool
tool1 = memory_tool.MemoryEstimate(program1, batch_size=32, name="program1")
tool2 = memory_tool.MemoryEstimate(program2, batch_size=32, name="program2")
tool.cal_multi_memories([tool1, tool2], serve=True, port=8111)
```
