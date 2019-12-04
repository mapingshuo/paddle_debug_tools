from paddle.fluid import core
import paddle.compat as cpt

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class MemoryEstimate(object):
  def __init__(self, main_program, batch_size=1, name=None):
    self.program = main_program
    # sub-block is not supported for now
    self.block = main_program.global_block() 
    self.debug_batchsize = batch_size
    self.name = name
    self.backward_start_idx = -1
    self.backward_finish_idx = -1
 
  def cal_memory(self, serve=False, port=8233):
      # analysis of the memory usage
      # cloned_memory_with_position = self.analysis_memory_usage(cloned_program.global_block())
      memory_with_position = self.analysis_memory_usage(self.block)
      self.draw(memory_with_position, serve, port)

  def _get_var_size(self, block, name, batch):
    """
    input:
        - block: where the var info is stored
        - name: var name
        - batch: batch size when training
    """
    dtype_to_size = {
            core.VarDesc.VarType.FP16: 2,
            core.VarDesc.VarType.FP32: 4,
            core.VarDesc.VarType.FP64: 8,
            core.VarDesc.VarType.INT16: 2,
            core.VarDesc.VarType.INT32: 4,
            core.VarDesc.VarType.INT64: 8,
            core.VarDesc.VarType.BOOL: 1,
            core.VarDesc.VarType.UINT8: 1,
    }
    if not block.desc.find_var(cpt.to_bytes(name)):
      print("get var size failed, can not find var name %s" % name)
      return 0
    #print(block.desc.var(cpt.to_bytes(name)))
    if block.desc.var(cpt.to_bytes(name)).type(
        ) != core.VarDesc.VarType.LOD_TENSOR:
      print("not lod tensor var, var name %s" % name)
      return 0
    var = block.var(name)
    if var.shape[0] == -1:
      res = -reduce(lambda x, y: x * y,
                          var.shape) * batch * dtype_to_size[var.dtype]
    else:
      res = reduce(lambda x, y: x * y,
                         var.shape) * dtype_to_size[var.dtype]
    if res < 0:
      return -res
    return res

  def draw(self, memory_with_position, serve, port, recompute_segments=None):
        #print("memory_with_position: ", memory_with_position)
        #print("recompute_segments", recompute_segments)

        x = [i - 2 for i in range(len(memory_with_position))]
        y = [(i * 1.0) / 1024 / 1024 for i in memory_with_position]

        if recompute_segments is not None:
            for i in recompute_segments:
                plt.axvline(x=i[1], color='r')
        
        x_margin = x[-1] * 0.05
        #print(max(y) * 0.1)
        if self.backward_start_idx != -1:
          plt.axvline(x=self.backward_start_idx, color='k', linestyle=':')
          plt.text(self.backward_start_idx - x_margin, max(y) * 0.5, 'forward propagation', rotation=90)
        if self.backward_finish_idx != -1:
          plt.axvline(x=self.backward_finish_idx, color='k', linestyle=':')
          plt.text(self.backward_finish_idx - x_margin, max(y) * 0.5, 'backward propagation', rotation=90)
          plt.text(x[-1] - x_margin, max(y) * 0.5, 'optimization', rotation=90)

        plt.plot(x, y, label="memory usage")
        #plt.plot(x2, y2, label="without_recompute")

        plt.legend(loc='upper right')
        plt.xlabel("op_idx")
        plt.ylabel('MB')
        plt.title('Estimated Memory Usage')
        plt.savefig('memory_anal.png')
        
        if serve is True:
          import SimpleHTTPServer
          import SocketServer

          Handler = SimpleHTTPServer.SimpleHTTPRequestHandler

          httpd = SocketServer.TCPServer(("", port), Handler)

          print("serving at port", port)
          httpd.serve_forever()

  def analysis_memory_usage(self, block):
    # find all vars()
    all_var_names = [x.name() for x in block.desc.all_vars()]
    #print("all_var_names")
    #for a in all_var_names:
    #    print(a)

    vars_create_position = {}
    vars_delete_position = {}
    for name in all_var_names:
      vars_create_position[name] = -1
      vars_delete_position[name] = -1

    for idx, op in enumerate(block.ops):
      #print(idx, op.type)
      for name in op.desc.input_arg_names():
        vars_delete_position[name] = idx
      for name in op.desc.output_arg_names():
        vars_delete_position[name] = idx
        if vars_create_position[name] == -1:
          vars_create_position[name] = idx

    for name in all_var_names:
      if block.var(name).persistable:
        vars_create_position[name] = -1
        vars_delete_position[name] = len(block.ops) - 1

        #for name in all_var_names:
        #    print(name, vars_create_position[name], vars_delete_position[name])

    position_to_var = {}
    for i in range(-1, len(block.ops)):
      position_to_var[i] = {"create": [], "delete": []}

    for name in all_var_names:
      position_to_var[vars_create_position[name]]['create'].append(name)
      position_to_var[vars_delete_position[name]]['delete'].append(name)

        #print('position_to_var: ', position_to_var)

    memory_timeline = 0
    memories = [0]
    for i in range(-1, len(block.ops)):
      #print(i, block.ops[i].type, memory_timeline)
      for var_name in position_to_var[i]['create']:
        if '@GRAD' in var_name and self.backward_start_idx == -1:
          self.backward_start_idx = i
        if '@GRAD' in var_name:
          self.backward_finish_idx = i
        #print('Create ', var_name, ', Size ',
        #self._get_var_size(block, var_name, self.debug_batchsize))
        memory_timeline += self._get_var_size(block, var_name,
                                                      self.debug_batchsize)
      memories.append(memory_timeline)
      for var_name in position_to_var[i]['delete']:
        #print('Delete ', var_name, ', Size ',
        #       self._get_var_size(block, var_name, self.debug_batchsize))
        memory_timeline -= self._get_var_size(block, var_name,
                                                      self.debug_batchsize)
            #memory_with_position.append(memory_timeline)
            #print(i, memory_timeline)

    return memories

def cal_multi_memories(memories, serve=False, port=8233):
    for mem_idx, memory in enumerate(memories): 
        memory_with_position = memory.analysis_memory_usage(memory.block)

        x = [i - 2 for i in range(len(memory_with_position))]
        y = [(i * 1.0) / 1024 / 1024 for i in memory_with_position]

        if mem_idx == 0:
          x_margin = x[-1] * 0.05
          #print(max(y) * 0.1)
          if memory.backward_start_idx != -1:
            plt.axvline(x=memory.backward_start_idx, color='k', linestyle=':')
            plt.text(memory.backward_start_idx - x_margin, max(y) * 0.7, 'forward propagation', rotation=90)
          if memory.backward_finish_idx != -1:
            plt.axvline(x=memory.backward_finish_idx, color='k', linestyle=':')
            plt.text(memory.backward_finish_idx - x_margin, max(y) * 0.7, 'backward propagation', rotation=90)
            plt.text(x[-1] - x_margin, max(y) * 0.7, 'optimization', rotation=90)
            plt.axvline(x=max(x), color='k', linestyle=':')

        if memory.name is not None: 
	    label = memory.name
        else:
	    label = "memory_usage_%d" % mem_idx
        plt.plot(x, y, label=label)
        #plt.plot(x2, y2, label="without_recompute")

    plt.legend(loc='upper right')
    plt.xlabel("op_idx")
    plt.ylabel('MB')
    plt.title('Estimated Memory Usage')
    plt.savefig('memory_anal.png')

    if serve is True:
      import SimpleHTTPServer
      import SocketServer

      Handler = SimpleHTTPServer.SimpleHTTPRequestHandler

      httpd = SocketServer.TCPServer(("", port), Handler)

      print("serving at port", port)
      httpd.serve_forever() 
