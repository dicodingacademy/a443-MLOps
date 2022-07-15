# TF-Serving Deployment use Docker

Membuat docker image 

```bash
docker build -t fashion-mnist-tf-serving .
```

Menjelankan docker image dan menentukan port. Ingat bahwa di dalam docker container TF Serving menggunakan port 8501.

```bash
docker run -p 8080:8501 fashion-mnist-tf-serving
```
