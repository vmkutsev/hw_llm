"""Independent CPU checks of the notebook's agent loop with scripted model output."""
import ast
import json
import re
from pathlib import Path
from smart_home import SMART_HOME_TOOLS, SYSTEM_PROMPT, SmartHome


def main():
    notebook = json.loads(Path(__file__).with_name('sft_function_calling_hw.ipynb').read_text())
    namespace = dict(json=json, re=re, SMART_HOME_TOOLS=SMART_HOME_TOOLS)
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type']=='code' and (source.startswith('TOOL_CALL_PATTERN') or source.startswith('def run_agent')):
            exec(compile(ast.parse(source), 'notebook_agent', 'exec'), namespace)
    def call(name, **arguments):
        return '<tool_call>'+json.dumps({'name':name,'arguments':arguments})+'</tool_call>'
    def run(outputs, max_steps=4):
        queue=iter(outputs)
        namespace['generate']=lambda *args,**kwargs: next(queue)
        home=SmartHome()
        messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':'scripted check'}]
        result=namespace['run_agent'](None,None,home,messages,max_steps=max_steps)
        return home,messages,result

    home,messages,result=run([call('turn_light',room='kitchen',state='on')+call('set_temperature',room='kitchen',value=23),'Done.'])
    assert home.snapshot()['rooms']['kitchen']=={'light':'on','temperature':23}
    assert [m['name'] for m in messages if m['role']=='tool']==['turn_light','set_temperature']
    assert result['stop_reason']=='final_answer'
    home,messages,result=run([call('get_room_state',room='bedroom'),call('set_temperature',room='bedroom',value=23),'Done.'])
    assert json.loads(messages[3]['content'])['temperature']==21
    assert home.snapshot()['rooms']['bedroom']['temperature']==23
    home,messages,result=run(['I cannot open the garage.'])
    assert home.snapshot()==SmartHome().snapshot() and not any(m['role']=='tool' for m in messages)
    home,messages,result=run(['<tool_call>{broken}</tool_call>'])
    assert result['stop_reason']=='invalid_tool_call' and home.snapshot()==SmartHome().snapshot()
    home,messages,result=run([call('turn_light',room='garage',state='on'),'That room is unsupported.'])
    assert 'error' in json.loads(messages[3]['content'])
    assert home.snapshot()==SmartHome().snapshot()
    home,messages,result=run([call('get_room_state',room='bedroom')],max_steps=1)
    assert result['stop_reason']=='max_steps' and len([m for m in messages if m['role']=='tool'])==1
    print('OK: parallel calls, sequential feedback, no-call, malformed JSON, simulator errors, step limit.')


if __name__=='__main__':
    main()
