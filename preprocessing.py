# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 18:08:07 2021

@author: egorb
"""
from json import load, loads
from os import listdir
from pandas import DataFrame
from datetime import datetime
from numpy import where
from plotly.express import box


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
    # Загрудка данных
    data = DataFrame()
    for channel in args:
        messages = []
        links = listdir(f'_{channel}')
        for link in links:
            file = read_json(f'_{channel}\\' + link)
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

# Словарь интервалов времени для удаления из текста
TIME_RANGES_DICT = {'5/2': '', }
for i in range(1,24):
    TIME_RANGES_DICT[r'[^0123456789]{0,1}' + str(i) + r':00'] = ' '
    TIME_RANGES_DICT[r'[^0123456789]{0,1}' + str(i) + r'\.00' + r'[^0123456789]{0,1}'] = ' '
    TIME_RANGES_DICT[r'[^0123456789]{0,1}' + str(i) + r':30'] = ' '
    TIME_RANGES_DICT[r'[^0123456789]{0,1}' + str(i) + r'\.30' + r'[^0123456789]{0,1}'] = ' '
    for j in range(i,24):
        TIME_RANGES_DICT[r'c[ \t]*?' + str(i) + r' до ' + str(j) + r'[^0123456789]{0,1}'] = ' '

# Словарь годов для удаления из текста
YEAR_RANGES_DICT = {str(k): '' for k in range(2001,2021,1)}


def replace_substring(row):
    new_string = row.clean_text
    for substring in row.substrings:
        new_string = new_string.replace(substring, '')
    return new_string

F_CURRENCY = {
    'usd': ['usd', '\$', 'dollar', 'доллар', 'бакс'],
    'eur': ['eur', '€', 'евро'],
    'gbp': ['£', 'gbp', 'фунт'],         
    }

TAX_DICT = {
    'gross': ['gros', 'грос', 'до налог', 'до вычет'],
    'net': ['net', 'после налог', 'после вычет', 'на руки', 'чисты']
    }


def apply_dict(df, cur, dict_to):
    select = df.index != df.index
    for pat in dict_to[cur]:
        select = select | df.clean_text.str.contains(pat)
    return select


def aprox_eng(df):
    return df.clean_text.str.count('a') > df.clean_text.str.count('а')
       
 
def get_currency(df):
    cur_col = where(df.forks.str.len() == 0, None,
              where(apply_dict(df, 'usd', F_CURRENCY), 'usd',
              where(apply_dict(df, 'eur', F_CURRENCY), 'eur',
              where(apply_dict(df, 'gbp', F_CURRENCY), 'gbp',
              where(aprox_eng(df), 'usd',
              where(df.forks.str[0].str[0] < 15, 'usd', 'rub'))))))
    return cur_col  


def get_tax(df):
    tax_col = where(df.forks.str.len() == 0, None,
              where(apply_dict(df, 'net', TAX_DICT), 'net',
              where(apply_dict(df, 'gross', TAX_DICT), 'gross',
              where(df.cur == 'rub', 'net', 'gross'))))
    return tax_col
    

def apply_tax(row):
    is_gross = row.tax == 'gross'
    forks = row.forks
    if not is_gross:
        return forks
    if row.cur != 'rub':
        return forks
    for i in range(len(forks)):
        forks[i] = [forks[i][0]*0.87, forks[i][1]*0.87]
    return forks


LEVELS_DICT = {
    'junior': {'pat': ['junior', 'джун', 'начинающ'], 'order': 0, 'q': (0.025, 0.85)},
    'middle': {'pat': ['middle', 'мидл', 'миддл'], 'order': 1, 'q': (0.05, 0.9)},
    'senior': {'pat': ['senior', 'синьор', 'сеньор', 'старш'], 'order': 2, 'q': (0.1, 0.95)},
    'lead': {'pat': ['lead', 'head', 'лид', 'ведущ', 'head'], 'order': 3, 'q': (0.15, 0.975)}
}


def add_levels_cols(df):
    df['junior'], df['middle'], df['senior'], df['lead'] = 0, 0, 0, 0
    
    for level in LEVELS_DICT.keys():
        for pat in LEVELS_DICT[level]['pat']:
            df[level] = df[level] | df.clean_text.str.contains(pat)
    
    return df


def level_fork(row, level):
    if not row[level]:
        return "[]"
    if len(row.forks) == 0:
        return "[]"
    order = LEVELS_DICT[level]['order'] if LEVELS_DICT[level]['order'] <= len(row.forks) - 1 else len(row.forks) - 1
    return str(row.forks[order])


def add_levels_forks(df):
    for level in LEVELS_DICT.keys():        
        df[level+'_fork'] = df.apply(lambda row: level_fork(row, level), axis=1)
        df[level+'_fork'] = df[level+'_fork'].apply(loads)
        
    return df


def format_forks(df):
    """Форматируем все вилки"""
    all_forks = []
    for i, row in df.iterrows():
        new_forks = []
        forks = row.forks
        for fork in forks:
            
            # Преобразуем вилки из одного числа к общему виду с границами
            if isinstance(fork, str):
                fork = (fork, fork)
                
            # Преобразуем границы из строк в числа
            new_fork = []
            for salary in fork:
                salary = salary.replace(',', '.').replace('\'', '.').replace(' ', '.')
                if len(salary) > 3 and salary.find('.') == -1 and salary[-2:] == '00':
                    salary = float(salary) / 1000
                else:
                    salary = float(salary)
                new_fork.append(salary)
            
            # Проверяем вилки на правую и левую границы и на уникальность
            if new_fork[0] <= new_fork[1] and new_fork not in new_forks:
                new_forks.append(new_fork)
            
        # Берем первые 3 вилки из каждой вакансии
        if len(new_forks) > 3:
            new_forks = new_forks[:4]
            
        # Сортируем вилки по возрастанию
        new_forks.sort()
            
        all_forks.append(new_forks)
         
    return all_forks


def select_level_only(df, level):
    other_levels = [i for i in LEVELS_DICT.keys() if i != level]
    level_select = df[level]
    for _level in other_levels:
        level_select = level_select & ~df[_level]
        
    return level_select


def level_quantiles(df, level, level_bounds={}):
    level_bounds[level] = {'bottom': [], 'upper': []}
    qb, qu = LEVELS_DICT[level]['q']
    level_select = select_level_only(df, level)
    level_bounds[level]['bottom'].append(round(df[level_select][level+'_fork'].str[0].quantile(qb), 0))
    level_bounds[level]['bottom'].append(round(df[level_select][level+'_fork'].str[0].quantile(qu), 0))
    
    level_bounds[level]['upper'].append(round(df[level_select][level+'_fork'].str[1].quantile(qb), 0))
    level_bounds[level]['upper'].append(round(df[level_select][level+'_fork'].str[1].quantile(qu), 0))
    
    print(f'Границы вилок для {level}: {level_bounds[level]}')
    
    return level_bounds


def get_quantiles(df):
    level_bounds = {}
    for level in LEVELS_DICT.keys():
        level_bounds = level_quantiles(df, level, level_bounds)
    return level_bounds


def levels_by_fork(fork, level_bounds):
    new_levels = []
    for level in level_bounds.keys():
        bottom = fork[0] >= level_bounds[level]['bottom'][0] and fork[0] <= level_bounds[level]['bottom'][1]
        upper = fork[1] >= level_bounds[level]['upper'][0] and fork[1] <= level_bounds[level]['upper'][1]
        if bottom and upper:
            new_levels.append(level)
            
    return new_levels
            
    
def estimate_level_forks(df, level_bounds):

    for i, row in df.iterrows():
        forks = row.forks
        if not forks:
            continue
        for fork in forks:
            for level in levels_by_fork(fork, level_bounds):
                df.loc[i, level+'_fork'] = str(fork)  
    return df


def box_plot_by_level(df):
    new_df = DataFrame()
    for level in LEVELS_DICT.keys():
        step_df = df[df[level + '_fork'].str.len()>0 & df[level]].loc[:, [level + '_fork']]
        step_df['Средняя зарплатная вилка'] = step_df[level + '_fork'].str[0]*0.5 + \
                                                    step_df[level + '_fork'].str[1]*0.5
        step_df['lvl'] = level
        new_df = new_df.append(step_df)
        new_df = new_df.append(step_df)
        
    fig = box(new_df, x="lvl", y="Средняя зарплатная вилка", color='lvl', labels={
                 "mean": "Уровень зарплат в тыс. рублях",
                 "lvl": 'Грейд'
             }, title = 'Средняя зарплатная вилка по грейдам')
    fig.show()


