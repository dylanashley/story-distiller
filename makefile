
all: __main__.py album_extractor.py features.py album_feature_encoder.pt templates.npz
	echo '#!/usr/bin/env python' > sdistil
	zip -j sdistil.zip __main__.py album_extractor.py features.py album_feature_encoder.pt templates.npz
	cat sdistil.zip >> sdistil
	rm sdistil.zip
	chmod +x sdistil

clean: sdistil
	rm -f sdistil

install: sdistil
	mkdir -p /usr/local/bin/
	cp sdistil /usr/local/bin/
