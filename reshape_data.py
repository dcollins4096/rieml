import tube_loader


data = tube_loader.load_many(check_file='tubes_take3.h5')
tube_loader.write_one('tubes_take3.h5',data)

