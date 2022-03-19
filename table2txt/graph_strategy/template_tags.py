class TemplateTag:
    #title = '[T]'
    
    #sub_name = '[SC]'
    #sub = '[S]'
    
    #obj_name = '[OC]'
    #obj = '[O]'

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
