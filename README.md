
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
tool = memory_tool.MemoryEstimate(fluid.default_main_program())
tool.cal_memory()
```

之后在相应文件夹下会生成一个memory_anal.png的文件，类似如下：

![example memory usage](image/memory_anal.png?raw=true "example memory usage")

上图是example/test_memory_tool.py的执行结果
