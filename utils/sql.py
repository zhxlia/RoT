import re
import sys
import sqlite3

sys.path.append('.')

from utils.table import Table
from typing import List, Dict


def fix_answer_sql(sql: str, table: List[List[str]]) -> str:
    sql = f"SELECT {sql.strip()}" if not sql.startswith("SELECT") else sql.strip()
    sql = sql.split(';')[0].replace('\n', ' ')
    # 将from中全部替换为information
    pattern = re.compile(
        r"FROM\s+.*?(?=\s+(WHERE|GROUP\s+BY|ORDER\s+BY|LIMIT)|$)", re.IGNORECASE)
    replaced_query = pattern.sub("FROM information", sql)
    # 右括号配对
    left_brucket = replaced_query.count('(')
    right_brucket = replaced_query.count(')')
    if left_brucket > right_brucket:
        replaced_query += ')' * (left_brucket - right_brucket)
    return replaced_query


def parse_answer_sql(sql: str, table: List[List[str]]) -> str:
    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()

    columns = ', '.join([f"{col} TEXT" for col in table[0]])
    creat_sent = str(Table(table, "information", "sql")).split("/*")[0].replace("\n", " ")
    # print(creat_sent)
    try:
        # cur.execute(f"CREATE TABLE information ({columns});")
        cur.execute(f"{creat_sent}")
    except Exception as e:
        print(f"table build error : {e}")
        return ""
    values_placeholder = ', '.join(['?'] * len(table[0]))
    rows = table[1:]
    try:
        cur.executemany(
            f"INSERT INTO information VALUES ({values_placeholder});", rows)
    except Exception as e:
        print(f"insert error : {e}")
        return ""

    try:
        cur.execute(sql)
        result = cur.fetchall()
    except Exception as e:
        return f"execute error of \" {sql} \" : {e}"
    finally:
        conn.close()

    if len(result) == 0:
        return ""
    answer = result[0]
    if isinstance(answer, tuple):
        answer = answer[0]
    return str(answer)


def extract_sql(sql: str) -> str:
    # unpack markdown format
    if '```sql' in sql:
        sql = sql.split('```sql')[-1].split('```')[0].strip()
    if '```' in sql:
        sql = sql.split('```')[1].strip()
    if '**SQL:**' in sql:
        sql = sql.split('**SQL:**')[-1].strip().strip('\n').strip('`')
    if 'SQL:' in sql:
        sql = sql.split('SQL:')[-1].strip().strip('\n').strip('`')
    sql = sql.split(';')[0]

    sql = sql.strip().strip(';').strip().strip("\n").strip()
    if not sql.lower().startswith('select'):
        sql = f"SELECT {sql}"
    sql = sql.replace('\n', ' ').replace('\t', ' ')

    # replace multiple spaces with single space
    sql = re.sub(r'\s+', ' ', sql)

    def fix_join(sql: str) -> str:
        # Helper function to replace commas with JOIN in a FROM clause
        def replace_commas(start: int) -> None:
            nonlocal sql
            paren_count = 0
            i = start
            while i < len(sql):
                if sql[i] == '(':
                    paren_count += 1
                elif sql[i] == ')':
                    paren_count -= 1
                # Replace comma with JOIN if not within parentheses
                elif sql[i] == ',' and paren_count == 0:
                    sql = sql[:i] + ' JOIN' + sql[i+1:]
                elif sql[i:i+4] == 'FROM' and paren_count == 0:
                    # Nested FROM, skip it
                    i += 4
                    continue
                elif sql[i:i+5] == 'WHERE' and paren_count == 0:
                    # Reached the end of the FROM clause
                    break
                i += 1

        # Main function logic
        i = 0
        while i < len(sql):
            # Find the FROM keyword
            from_index = sql.find('FROM', i)
            if from_index == -1:
                break  # No more FROM clauses found
            # Call the helper to replace commas starting from the end of 'FROM '
            replace_commas(from_index + 5)
            i = from_index + 5  # Update the index to continue searching

        return sql

    sql = fix_join(sql)

    return sql

