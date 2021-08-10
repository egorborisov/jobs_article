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
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go


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
    'Junior': {'pat': ['junior', 'джун', 'начинающ'], 'order': 0, 'q': (0.025, 0.85)},
    'Middle': {'pat': ['middle', 'мидл', 'миддл'], 'order': 1, 'q': (0.05, 0.9)},
    'Senior': {'pat': ['senior', 'синьор', 'сеньор', 'старш'], 'order': 2, 'q': (0.1, 0.95)},
    'Lead': {'pat': ['lead', 'head', 'лид', 'ведущ', 'head'], 'order': 3, 'q': (0.15, 0.975)}
}

def add_levels_cols(df):
    df['Junior'], df['Middle'], df['Senior'], df['Lead'] = 0, 0, 0, 0
    
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
                 "lvl": 'Уровень'
             }, title = 'Средняя зарплатная вилка по грейдам')
    fig.show()


GRADES = ['Junior', 'Middle', 'Senior', 'Lead']


def get_salaries_by_categories(df, categories):
    dfs = []
    for category in categories:
        for grade in GRADES:
            salary_column = grade + '_fork_avg'
            cur_df = df[(df['cur'] == 'rub') & (df[category]) & (df[grade]) & (df[salary_column])]
            cur_df = pd.DataFrame(data={'salary': cur_df[salary_column]})
            cur_df['category'] = category
            cur_df['grade'] = grade
            dfs.append(cur_df)
    salary = pd.concat(dfs).reset_index()
    return salary


def get_avg_salaries(salary):
    avg_salary = salary.groupby(['grade', 'category']).mean().reset_index()
    avg_salary = avg_salary.sort_values(by=['grade'], key=lambda x: x.map({'Junior': 1, 'Middle': 2, 'Senior': 3, 'Lead': 4}))
    return avg_salary


NORMAL_FORKS = {
    'Junior': {'lower': [30, 100], 'upper': [40, 150]},
    'Middle': {'lower': [87, 174], 'upper': [139, 250]},
    'Senior': {'lower': [100, 250], 'upper': [167, 400]},
    'Lead': {'lower': [100, 350], 'upper': [150, 500]}
}


def lower_fork(fork, normal_fork):
    return np.nan if not normal_fork or len(fork) == 0 else min(fork)

def avg_fork(fork, normal_fork):
    return np.nan if not normal_fork or len(fork) == 0 else (min(fork) + max(fork)) / 2

def upper_fork(fork, normal_fork):
    return np.nan if not normal_fork or len(fork) == 0 else max(fork)

def fork_diff(fork, normal_fork):
    return np.nan if not normal_fork or len(fork) == 0 else max(fork) - min(fork)

def fork_ratio(fork, normal_fork):
    return np.nan if not normal_fork or len(fork) == 0 else max(fork) / min(fork)

def fork_coef(fork, normal_fork):
    return np.nan if not normal_fork or len(fork) == 0 else (max(fork) - min(fork)) / min(fork)

def normal_fork(fork, grade):
    if len(fork) == 0:
        return False
    return NORMAL_FORKS[grade]['lower'][0] <= fork[0] <= NORMAL_FORKS[grade]['lower'][1] and\
           NORMAL_FORKS[grade]['upper'][0] <= fork[1] <= NORMAL_FORKS[grade]['upper'][1]


def get_keyword_df(df, keywords, include_spaces=True):
    for keyword, keywords_to_search in keywords.items():
        keywords_regex = '|'.join(keywords_to_search)
        if include_spaces:
            keywords_regex = '\W(' + keywords_regex + ')\W'
        df[keyword] = df['clean_text'].str.contains(keywords_regex, regex=True)
    keywords_df = pd.DataFrame(columns=['keyword', 'number'])
    for keyword in list(keywords.keys()):
        keywords_df = keywords_df.append({'keyword': keyword, 'number': len(df[df[keyword]])}, ignore_index=True)
    keywords_df = keywords_df.sort_values(by=['number'], ascending=False).reset_index(drop=True)
    keywords_df['perc'] = keywords_df['number'] / len(df) * 100
    return keywords_df


def plot_top_keywords(keyword_df, top_n, title, xlabel, ylabel, palette='summer_r', scale_perc=False):
    ax = sns.barplot(data=keyword_df[keyword_df.index < top_n], x='keyword', y='perc', palette=palette)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if scale_perc:
        ax.set_ylim(0, 100)


def plot_skill_stats(df):
    layout = dict(plot_bgcolor='white',
                margin=dict(t=20, l=20, r=20, b=20),
                xaxis=dict(title='Процент вакансий',
                            linecolor='#d9d9d9',
                            showgrid=False,
                            mirror=True,
                            range=(0, 100)),
                yaxis=dict(title='Ценность навыка',
                            linecolor='#d9d9d9',
                            showgrid=False,
                            mirror=True))

    data = go.Scatter(
        x=df['perc'],
        y=df['weight'],
        text=df['skill'],
        textposition='top center',
        textfont=dict(color='black'),
        mode='markers+text',
        marker=dict(color='#B533FF', size=8),
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def count_reactions(dicts):
    reactions = 0
    for dict_ in dicts:
        reactions += len(dict_['users'])
    return reactions


def assign_reactions(df, reactions_df, top_n):
    top_n = 100
    top_reactions = list(reactions_df.loc[:top_n]['name'])
    for index, row in df.iterrows():
        reaction_dicts = df.loc[index, 'reactions']
        if isinstance(reaction_dicts, float):
            continue
        for top_reaction in top_reactions:
            df['reaction' + top_reaction] = 0
            reaction_found = False
            for reaction_dict in reaction_dicts:
                if reaction_dict['name'] == top_reaction:
                    df.loc[index, 'reaction_' + top_reaction] = int(reaction_dict['count'])
                    reaction_found = True
                    break
            if not reaction_found:
                df.loc[index, 'reaction_' + top_reaction] = 0
        df['reaction_' + top_reaction] = df['reaction_' + top_reaction].fillna(0).astype(np.int32)































"""
ukraine_cities = {
    'Киев': ['kiy?ev', 'киев[ае]?'],
    'Харьков': ['kharkov', 'харьков[ае]?'],
    'Одесса': ['odessa', 'одесс[аеы]'],
    'Днепр': ['dnipro', 'днепропетровск[ае]?'],
    'Львов': ['lviv', 'львов[ае]?'],
}

belarus_cities = {
    'Минск': ['minsk', 'минск[ае]?'],
    'Гомель': ['gomel', 'гомел[ьи]'],
    'Витебск': ['vitebsk', 'витебск[ае]?'],
    'Брест': ['брест[ае]?'],
    'Могилёв': ['mogilev', 'могил[её]в[ае]?']
}

england_cities = {
    'Лондон': ['london', 'лондон[ае]?'],
    'Ливерпуль': ['liverpool', 'ливерпул[ьея]'],
    'Манчестер': ['manchester', 'манчестер[ае]?'],
    'Лестер': ['leicester', 'лестер[ае]?'],
    'Ньюкасл': ['newcastle', 'ньюкасл[ае]?'],
    'Оксфорд': ['oxford', 'оксфорд[ае]?'],
    'Лидс': ['leeds', 'лидс[ае]?'],
    'Бирминген': ['birmingen', 'б[еёи]рмингем[ае]?']
}

france_cities = {
    'Париж': ['paris', 'париж[ае]?'],
    'Лион': ['lyon', 'лион[ае]?'],
    'Монако': ['monaco', 'монако'],
    'Ниица': ['in nice', 'ницц[аеы]'],
    'Марсель': ['marsei?lle', 'марсел[ьея]'],
    'Тулуза': ['toulouse', 'тулуз[аеы]'],
    'Страсбург': ['strasbo?urg', 'страсбург[ае]'],
    'Бордо': ['bordeaux', 'бордо'],
    'Монпелье': ['montpellier', 'монпелье'],
    'Ренн': ['rennes', 'ренн[ае]'],
    'Реймс': ['reims', 'р[еэ]ймс[ае]'],
    'Дижон': ['dijon', 'дижон[ае]'],
    'Канн': ['cannes', 'канн?(а|е|ах)?']
}

germany_cities = {
    'Мюнхен': ['munchen', 'мюнхен[ае]?'],
    'Дортмунд': ['dortmund', 'дортмунд[ае]?'],
    'Берлин': ['berlin', 'берлин[ае]?'],
    'Дрезден': ['dre[sz]den', 'др[еэ]зд[еэ]н[ае]?'],
    'Франкфурт': ['frankfurt', 'франкфурт[ае]?'],
    'Гамбург': ['hamburg', 'гамбург[ае]?'],
    'Штутгарт': ['stuttgart', 'штутт?гарт[ае]?'],
    'Бремен': ['bremen', 'бремен[ае]?'],
    'Лейпциг': ['leipzig', 'ле[ий]пциг[ае]?']
}

italy_cities = {
    'Рим': ['rome', 'рим[ае]?'],
    'Милан': ['milan', 'милан[ае]?'],
    'Турин': ['turin', 'турин[ае]?'],
    'Флоренция': ['florence', 'флоренци[ия]'],
}

spain_cities = {
    'Барселона': ['barcelona', 'барселон[аеы]'],
    'Мадрид': ['madrid', 'мадрид[ае]?'],
    'Севилья': ['sevilla', 'севиль[еия]'],
    'Бильбао': ['bilbao', 'бильбао'],
    'Валенсия': ['valencia', 'валенси[ия]'],
}

usa_cities = {
    'Нью-Йорк': ['new\Wyork', 'нью\Wйорк[ае]?'],
    'Лос-Анджелес': ['los\Wangeles', 'лос\Wанджелес[ае]?'],
    'Вашингтон': ['washington', 'вашингтон[ае]?'],
    'Сан Франциско': ['san\Wfrancisco', 'сан\Wфранциско'],
    'Сеаттл': ['seattle', 'сеатт?л[ае]?'],
    'Чикаго': ['chicago', 'чикаго'],
    'Калифорния': ['california', 'калифорни[еия]'],
    'Бостон': ['boston', 'бостон[ае]?'],
    'Денвер': ['denver', 'денвер[ае]?'],
    'Майами': ['miami', 'майами']
}


def flatten(lst):
    return [item for sublist in lst for item in sublist]


countries = {
    # Европа
    'Россия': ['russia', 'росси[ия]', 'рф'] + flatten(list(russia_cities.values())),
    'Украина': ['ukraine', 'украин[аеы]'] + flatten(list(ukraine_cities.values())),
    'Белоруссия': ['belarus', 'белорусси[ия]', 'бел[оа]рус[ьи]и?'] + flatten(list(belarus_cities.values())),
    
    'Англия': ['england', 'united kingdom', '\Wuk\W', 'англи[ия]'] + flatten(list(england_cities.values())),
    'Франция': ['france', 'франци[ия]'] + flatten(list(france_cities.values())),
    'Германия': ['germany', 'германи[ия]'] + flatten(list(germany_cities.values())),
    'Италия': ['italy', 'итали[ия]'] + flatten(list(italy_cities.values())),
    'Испания': ['spain', 'испани[ия]'] + flatten(list(spain_cities.values())),
    
    'Швейцария': ['switzerland', 'швейцари[ия]'],
    'Швеция': ['sweden', 'швеци[ия]'],
    'Нидерланды': ['netherlands', 'нидерланд(ы|ах)', 'голланди[ия]', 'amsterdam', 'амстердам'],
    'Австрия': ['austria', 'австри[ия]'],
    'Греция': ['greece', 'греци[ия]'],
    'Турция': ['turkey', 'турци[ия]'],
    'Кипр': ['cyprus', 'кипр[ае]?'],
    'Грузия': ['georgia', 'грузи[ия]'],
    'Израиль': ['israel', 'израил[ьея]'],
    'Финляндия': ['finland', 'финлянди[ия]'] + ['helsinki', 'хель?синк'],
    'Эстония': ['estonia', 'эстони[ия]'],
    'Литва': ['литв[аеы]'],
    'Латвия': ['латви[ия]'],
    
    # Азия
    'Китай': ['china', 'кита[ейя]'],
    'Корея': ['korea', 'коре[еия]'],
    'ОАЭ': ['uae', 'united arab emirates', 'оаэ', 'арабски[ех] эмират(ы|ах)'],
    'Индия': ['india', 'инди[ия]'],
    'Катар': ['qatar', 'катар[ае]?'],
    
    # Северная Америка
    'Канада': ['canada', 'канад[аеы]'],
    'США': ['usa', 'united states', 'сша', 'в штатах'] + flatten(list(usa_cities.values())),
    'Мексика': ['mexico', 'мексик[аеи]'],
    
     # Австралия
    'Австралия': ['australia', 'автрали[ия]'],
}

keyword_df = get_keyword_df(df, countries, include_spaces=True)
df['number_of_skills'] = df[skills].sum(axis=1)


# анализ по компаниям

companies = {
    'Сбербанк': ['sber', 'sberbank', 'сбер', 'сбербанк'],
    'ВТБ': ['vtb', 'втб'],
    'Райффайзен': ['raiffeisen', 'райффайзен'],
    'Тинькофф': ['tinkoff', 'тинькофф'],
    'Альфа-Банк': ['альфа', 'alpha.?bank'],
    'Точка банк': ['tochka.com', '(tochka|точка)\W(bank|банк)'],
    'Банк спб': ['(bank|банк)\W(spb|спб)', 'банк\w?\Wсанкт\W?петербург'],
    'Яндекс': ['яндекс', 'yandex'],
    'Вконтакте': ['\Wvk\W', 'vkontakte', '\Wвк\W', 'вконтакте'],
    'Одноклассники': ['одноклассники'],
    'X5 Retail Group': ['x5', 'retail.?group', 'х5'],
    'Mail.ru Group': ['\W(mail|м[аеэ]йл)\W'],
    'Huawei': ['huawei', 'хуав[еэ]й'],
    'JetBrains': ['jet.?brains', 'jb', 'д?жет.?бр[аеэ]йнс', 'жб'],
    'EPAM': ['epam', '[эе]пам'],
    'Rambler Group': ['rambler', 'рамблер'],
    'МТС': ['мтс'],
    'Билайн': ['beeline', 'biline', 'билайн'],
    'Мегафон': ['мегафон', 'megafon'],
    'Tele 2': ['tele.?2', 'теле.?2'],
    'Ростелеком': ['rostelekom', 'ростелеком'],
    'S7': ['s7'],
    'Газпром': ['gazprom', 'газпром'],
    'Ozon': ['ozon', 'озон'],
    'Авито': ['avito', 'авито'],
    'Delivery Club': ['delivery.?club'],
    'Wildberries': ['wildberries'],
    'Leroy Merlin': ['ler[uo][ay].?merl[ei]n', 'леруа.?мерлен'],
    'Лента': ['lenta', 'лента']
}

keyword_df = get_keyword_df(df, companies, include_spaces=False)
top_n = 5
top_n_list = list(keyword_df['keyword'])[:top_n]
salary = get_salaries_by_categories(df, top_n_list)
fig = px.box(salary, x='category', y='salary', color='grade', title='Средняя зарплата по компаниям')
fig.show()




# сферы применения Data Science
spheres = {
    'Marketing': {'marketing', 'commerce', 'advertisment', 'маркетинг', 'коммерци', 'продви[жг]\w{,4}.продукт'},
    'Banking': {'fraud', '(credit|data.?risk)', 'risk modeling', 'фрод', 'кредит', 'микрофинанс', 'мфо', 'риск', 'займ'},
    'Finances': {'cryptocurrency', 'quantative', 'trading', 'stocks', 'bitcoin', 'крипт[ао]', 'тр[еэ]й?динг', 'акци', 'битко[ий]н'},
    'Retail': {'retail', 'р[еи]т[еэ]йл', '(супер|гипер)маркет', '[хx]5', 'еда', 'напитки'},
    'Medicine': {'medicine', 'medical', 'biology', 'health', 'illness', 'drug', 'pharmac', 'bioinformatic'
                 'медицин', 'биологи', 'здоровь', 'болезн', 'наркотик', 'аптек', 'биоинформатик'},
    'Gaming': {'gaming', 'game analysis', 'gambling', 'casino', 'игров', 'aнализ игр', 'гейм', 'г[аеэ]мблинг', 'казино', 'матч', 'турнир'},
    'Transport': {'transport', 'driv(ing|er)', 'cars', 'passengers', 'autopilot', 'plane',
                  'транспорт', 'вождение', 'водител', 'пассажир', 'автопилот', 'самол[её]т', 'передви[жг]'}
}

get_keyword_df(df, spheres, include_spaces=False)
salary = get_salaries_by_categories(df, list(spheres.keys()))
fig = px.box(salary, x='category', y='salary', color='grade', title='Средняя зарплата по различным категориям')
fig.show()
"""