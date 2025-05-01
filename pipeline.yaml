pipeline {
    agent any

    environment {
        DOCKER_REGISTRY = 'shivampandey88'
        BACKEND_IMAGE = "${DOCKER_REGISTRY}/backend:latest"
        FRONTEND_IMAGE = "${DOCKER_REGISTRY}/frontend:latest"
    }

    stages {
        stage('Checkout Code') {
            steps {
                git url: 'https://github.com/shivampandey-14/Spe-major.git', branch: 'main'
            }
        }

        stage('Clean Docker System And Kind Cluster') {
            steps {
                echo 'Cleaning up unused Docker data...'
                sh 'docker system prune -a -f'
                echo 'Cleaning up Kind cluster...'
                sh 'kind delete cluster --name kind-cluster || true'
            }
        }

        stage('Build Docker Images') {
            steps {
                echo 'Building Docker images...'
                sh '''
                    docker build -t ${BACKEND_IMAGE} ./Backend
                    docker build -t ${FRONTEND_IMAGE} ./Frontend
                '''
            }
        }

        stage('Push Docker Images to Docker Hub') {
            steps {
                echo 'Pushing images to Docker Hub...'
                sh '''
                    docker push ${BACKEND_IMAGE}
                    docker push ${FRONTEND_IMAGE}
                '''
            }
        }

        stage('Create Kind Cluster') {
            steps {
                echo 'Creating Kind cluster...'
                sh '''
                    kind delete cluster --name kind-cluster || true

                    cat <<EOF > kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30501
        hostPort: 30501
      - containerPort: 30500
        hostPort: 30500
EOF

                    kind create cluster --name kind-cluster --config=kind-config.yaml
                    kind export kubeconfig --name kind-cluster --kubeconfig=${WORKSPACE}/kind-kubeconfig.yaml
                '''
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                echo 'Deploying backend and frontend...'
                sh '''
                    export KUBECONFIG=${WORKSPACE}/kind-kubeconfig.yaml
                    kubectl apply -f ./k8s/backend-deployment.yaml 
                    kubectl apply -f ./k8s/backend-service.yaml 
                    kubectl apply -f ./k8s/frontend-deployment.yaml 
                    kubectl apply -f ./k8s/frontend-service.yaml 
                '''
            }
        }

        stage('Wait for Deployments to be Ready') {
            steps {
                echo 'Waiting for app to be ready...'
                sh '''
                    export KUBECONFIG=${WORKSPACE}/kind-kubeconfig.yaml
                    kubectl rollout status deployment/backend-deployment
                    kubectl rollout status deployment/frontend-deployment
                '''
            }
        }

        stage('Test Service Access') {
            steps {
                echo 'Testing frontend and backend services...'
                sh '''
                    export KUBECONFIG=${WORKSPACE}/kind-kubeconfig.yaml

                    echo "Testing Frontend on localhost:30500"
                    curl --fail http://localhost:30500 || echo "Frontend not reachable"

                    echo "Testing Backend on localhost:30501"
                    curl --fail http://localhost:30501 || echo "Backend not reachable"
                '''
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished!'
        }
    }
}
