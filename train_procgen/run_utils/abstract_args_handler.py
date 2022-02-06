class AbstractArgsHandler(object):
    '''
        Base arguments handler.
    '''
    @staticmethod
    def set_parser_arguments(parser):
        '''
            Add argument options to parser.
        '''
        pass

    @staticmethod
    def get_object_from_arguments(args):
        '''
            Parse argurments.
        '''
        pass