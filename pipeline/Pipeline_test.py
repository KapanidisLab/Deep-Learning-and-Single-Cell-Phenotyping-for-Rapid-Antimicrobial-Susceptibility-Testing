
class Test_ProcessingPipeline:


    def test_ObjectFactory(self):
        from ProcessingPipeline import ObjectFactory

        def ExampleF():
            return 1

        Factory = ObjectFactory()
        Factory.register_implementation(('key1','key2'), ExampleF)
        function = Factory._create(('key1','key2'))

        value = function()
        assert value == 1

    def test_ProcessingPipeline(self):
        from ProcessingPipeline import ProcessingPipeline

        def dummy_sorter(path):
            return path
        def dummy_collector(path):
            return path

        #TODO add testing for ImageOp, generate some dummy data to read from file.

        data_folder = './Somewhere/Something/'
        Pipeline = ProcessingPipeline(data_folder, 'dummy')

        Pipeline._Factory.register_implementation(('sorter','dummy'), dummy_sorter)
        Pipeline._Factory.register_implementation(('collector','dummy'), dummy_collector)

        Pipeline.Sort()
        Pipeline.Collect()

        #Check return flags
        assert Pipeline.sorted == True
        assert Pipeline.collected == True

class Test_helpers:

    #TODO add testing for (dir and file) counter, generate dummy directory tree for them to traverse.

    def test_interspread(self):
        from helpers import interspread

        string = ['Nice','To','Meet','You']
        separator = '_'
        target = 'Nice_To_Meet_You'

        assert interspread(string,separator) == target

    def test_makedir(self):
        from helpers import makedir
        import os
        dir = "./dummydir"

        try:
            makedir(dir)
            assert os.path.exists(dir)

        finally:
            os.rmdir(dir) #rmdir removes empty dirs only, so is safe

    def test_im_2_uint16(self):
        from helpers import im_2_uint16
        import numpy

        # 1 test cases, uint8 images, maybe add more.

        im_uint8 = numpy.arange(0,256,1)
        im_uint8 = numpy.tile(im_uint8,(256,1))
        im_uint8 = numpy.asarray(im_uint8,dtype='uint8')

        target_uint8 = numpy.asarray(im_uint8 * (65535/255), dtype='uint16')

        assert (im_2_uint16(im_uint8) == target_uint8).all() and  im_2_uint16(im_uint8).dtype == target_uint8.dtype


    #TODO add testing class for implementations, need to generate dummy data for that.

