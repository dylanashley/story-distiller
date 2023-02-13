
all: __main__.py extractor.py features.py model.pt
	echo '#!/usr/bin/env python' > sdi
	zip sdi.zip __main__.py extractor.py features.py model.pt
	cat sdi.zip >> sdi
	rm sdi.zip
	chmod +x sdi

clean: sdi
	rm -f sdi

install: sdi
	mkdir -p /usr/local/bin/
	cp sdi /usr/local/bin/
