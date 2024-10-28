kfp:
	docker build -t footprintai/kfp-compiler:0.1.0 \
		--no-cache -f Dockerfile.kfp .
	docker push footprintai/kfp-compiler:0.1.0

gen-manifests:
	docker run -v $(PWD):/app \
		--entrypoint /app-script/gen-manifests.sh \
		footprintai/kfp-compiler:0.1.0
