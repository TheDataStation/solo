class SqlQuery:
    sel_op = 'SELECT'
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    where_op = 'WHERE'
    and_op = 'AND' 

    @staticmethod
    def get_meta_tags():
        sql_word_lst = []
        sql_word_lst.append(SqlQuery.sel_op)
        for op in SqlQuery.agg_ops:
            if op != '':
                sql_word_lst.append(op)
        for op in SqlQuery.cond_ops:
            if op != 'OP':
                sql_word_lst.append(op)
        sql_word_lst.append(SqlQuery.where_op)
        sql_word_lst.append(SqlQuery.and_op)
        
        meta_tags = []
        for sql_word in sql_word_lst:
            tag = SqlQuery.get_src_tag(sql_word)
            meta_tags.append(tag)
        return meta_tags
         
    @staticmethod
    def get_src_tag(sql_word):
        if sql_word == '=':
            word = 'EQ'
        elif sql_word == '>':
            word = 'GT'
        elif sql_word == '<':
            word = 'LT'
        elif sql_word == 'OP':
            raise ValueError('[%s] not supported' % sql_word)
        else:
            word = sql_word 
             
        chrs = [a.upper() for a in word]
        tag = '[' + '-'.join(chrs) + ']'
        return tag

