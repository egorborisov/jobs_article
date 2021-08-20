from datetime import datetime
from json import load
from os import listdir
from pandas import DataFrame


def read_json(filename):
    with open(filename, encoding='utf-8') as file:
        return load(file)
    
def convert_ts(ts):
    if isinstance(ts, float) and ts == ts and ts is not None:
        return datetime.fromtimestamp(ts)
    else:
        return None
        
def load_messages(*args):
    """Загружаем все сообщения в pandas DataFrame и убираем технические записи"""
    # Загрузка данных
    data = DataFrame()
    for channel in args:
        messages = []
        links = listdir(f'_{channel}')
        for link in links:
            file = read_json(f'_{channel}/' + link)
            messages = messages + file
        messages = DataFrame(messages)
        messages['source'] = channel
        data = data.append(messages, sort=False)
    data = data.reset_index(drop=True)
    print(f'Загружено {len(data)} cообщений')
    
    # Убираем технические сообщения
    select = (data.upload.isna()) & (data.display_as_bot.isna()) & data.subtype.isna()
    data = data[select]
    print(f"\tУдалены технические сообщения: {sum(~select) } сообщений")
    
    # Оставляем только выбранные колонки 
    cols = ['ts', 'user', 'text', 'user_profile', 'attachments', 'edited',
           'reactions', 'thread_ts', 'reply_count', 'reply_users_count',
           'latest_reply', 'reply_users', 'replies', 'subscribed',
           'parent_user_id', 'last_read', 'client_msg_id', 'blocks', 'old_name',
           'name', 'source']
    
    # Меняем формат для дат
    data['ts'] = ((data['ts'].astype('float')).apply(convert_ts)).dt.round('s')
    data['thread_ts'] = ((data['thread_ts'].astype('float')).apply(convert_ts)).dt.round('s')
    print(f"\tДата и время округлены до секунд и переведены в datetime формат")
    
    # Очищаем и предобрабатываем текст
    data['text'] = data['text'].str.lower().str.strip()
    data['text'] = data.text.str.replace('\xa0', ' ')
    print(f"\tТекст приведен к маленьким буквам и очищен от лишних пробелов")
                    
    print(f"Осталось {len(data)} cообщений")
    
    return data.loc[:,cols].reset_index(drop=True)