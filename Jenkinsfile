pipeline {
    agent any

    parameters {
        string(name: 'namespace', defaultValue: "mom-server", description: 'namespace to deploy')
    }
    environment {
        // Access environment variables using Jenkins credentials
        DOCKER_REGISTRY = credentials('momentum-server-docker-registry')
        GKE_CLUSTER = credentials('mom-core-gke-cluster')
        GKE_ZONE = credentials('gke-zone')
        GCP_PROJECT = credentials('gcp-project')
        GOOGLE_APPLICATION_CREDENTIALS = credentials('google-application-credentials')
    }

    stages {
        stage('Checkout') {
            steps {
                script {
                    // Determine environment based on branch
                    def branch = env.GIT_BRANCH

                    if (branch == "origin/temp") {
                        env.ENVIRONMENT = 'temp'
                    } else if (branch == "origin/main") {
                        env.ENVIRONMENT = 'main'
                    } else if (branch == "origin/devops"){
                        env.ENVIORNMENT = 'devops'
                    } else {
                        error("Unknown branch: ${branch}. This pipeline only supports main and staging branches.")
                    }

                    checkout scm
                    // Capture the short Git commit hash to use as the image tag
                    env.GIT_COMMIT_HASH = sh(returnStdout: true, script: 'git rev-parse --short HEAD').trim()
                }
            }
        }

        stage('Configure Docker Authentication') {
            steps {
                script {
                    // Extract the registry's hostname for authentication
                    def registryHost = env.DOCKER_REGISTRY.tokenize('/')[0]
                    sh """
                        sudo gcloud auth configure-docker ${registryHost}
                    """
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    // Use the Git commit hash as the image tag
                    def imageTag = env.GIT_COMMIT_HASH
                    def dockerRegistry = env.DOCKER_REGISTRY
                    echo "Printing the saved docker registry from env:"
                    echo "${dockerRegistry}"
                    sh "sudo docker build -t ${DOCKER_REGISTRY}/momentum-server:${imageTag} ."
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    // Use the Git commit hash as the image tag
                    def imageTag = env.GIT_COMMIT_HASH
                    echo "printing the user here"
                    sh "whoami && pwd"
                    sh "sudo docker push ${DOCKER_REGISTRY}/momentum-server:${imageTag}"
                }
            }
        }

        stage('Configure GKE Authentication') {
            steps {
                script {
                    // Use the service account path from credentials
                    sh """
                    sudo gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                    sudo gcloud container clusters get-credentials ${GKE_CLUSTER} --zone ${GKE_ZONE} --project ${GCP_PROJECT}
                    """
                }
            }
        }

         stage('Ask User for Deployment Confirmation') {
            steps {
                script {
                    def deployConfirmation = input(
                        id: 'userInput',
                        message: 'Do you want to deploy the new Docker image?',
                        parameters: [
                            choice(name: 'Deploy', choices: ['Yes', 'No'], description: 'Select Yes to deploy the image or No to abort.')
                        ]
                    )

                    if (deployConfirmation == 'No') {
                        error('User chose not to deploy the images, stopping the pipeline.')
                    }
                }
            }
        }

        stage('Deploy Image') {
            steps {
                script {
                    def imageDeploySucceeded = false
                    def imageTag = env.GIT_COMMIT_HASH

                    echo "this is the fetched docker image tag: ${imageTag}"


                    try {
                        sh """
                        kubectl set image deployment/momentum-server-deployment momentum-server=${DOCKER_REGISTRY}/momentum-server:${imageTag} -n ${params.namespace}
                        kubectl rollout status deployment/momentum-server-deployment -n ${params.namespace}
                        """
                        imageDeploySucceeded = true
                    } catch (Exception e) {
                        echo "Deployment failed: ${e}"
                    }

                    if (!imageDeploySucceeded) {
                        echo 'Rolling back to previous revision...'
                        sh 'kubectl rollout undo deployment/momentum-server-deployment -n ${params.namespace}'
                    }
                }
            }
        }



        stage('Pipeline finished') {
            steps {
                script {

                    echo "Pipeline finished"
                    // Check the deployment status
                    sh """
                    echo "checking the deployment status" && kubectl get pods -n ${params.namespace}
                    """

                }
            }
        }

    }

    post {
        always {
            echo "Pipeline finished"
            // Optional cleanup action
            script {

                // Clean up local Docker images
                def imageTag = env.GIT_COMMIT_HASH
                sh """
                docker rmi ${DOCKER_REGISTRY}/momentum-server:${imageTag} || true
                """
            }
        }
    }
}
