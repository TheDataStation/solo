class RelationTag:
    tag_title = '[T]'
    tag_sub_name = '[SC]'
    tag_sub = '[S]'
    tag_obj_name = '[OC]'
    tag_obj = '[O]'

    sub_none = ''
    obj_none = ''

    @staticmethod
    def get_annotated_text(title, sub_name, sub, obj_name, obj):
        assert (title is not None)
        if sub_name is None:
            sub_name = ''
        if sub is None:
            sub = ''
        assert (obj_name is not None)
        assert(obj is not None)

        sub_part = sub_name + '  ' + sub
        obj_part = obj_name + '  ' + obj
        out_text = title + '  ,  ' + sub_part.strip() + '  ' + obj_part.strip() + ' . '

        return out_text
    
    @staticmethod
    def get_tagged_text(title, sub_name, sub, obj_name, obj):
        assert (title is not None)
        if sub_name is None:
            sub_name = ''
        if sub is None:
            sub = ''
        assert (obj_name is not None)
        assert(obj is not None)

        out_text = '%s %s %s %s %s %s %s %s %s %s' % (RelationTag.tag_title, title, 
                                                      RelationTag.tag_sub_name, sub_name, 
                                                      RelationTag.tag_sub, sub, 
                                                      RelationTag.tag_obj_name, obj_name, 
                                                      RelationTag.tag_obj, obj)

        return out_text
