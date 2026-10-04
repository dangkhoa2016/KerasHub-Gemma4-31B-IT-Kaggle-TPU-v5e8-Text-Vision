# Giới hạn

- Target đã qualification là Gemma 4 31B Instruct trên Kaggle TPU v5e-8 với 8
  TPU devices. Accelerator khác không nằm trong cùng qualification.
- Chưa hỗ trợ audio generation.
- Hot latency phụ thuộc shape, generation bucket, cache state và khả năng reuse
  compilation cache. Con số 5,768 giây của identical hot request không phải SLA cho mọi request.
- Load model và compile đầu tiên có thể mất nhiều phút.
- Job store nằm trong memory của process, không phải distributed durable queue.
- Quick Tunnel phù hợp demo, không phải ingress production lâu dài.
- Model availability/license tuân theo upstream Gemma.
- Không cam kết tương thích với mọi phiên bản Keras/KerasHub/JAX/libtpu tương lai.
