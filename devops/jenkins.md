
最基本的连续交付流程将至少具有三个阶段，这些阶段应在以下内容中定义Jenkinsfile：**构建**，**测试**和**部署**。

```sh
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                echo 'Building'
            }
        }
        stage('Test') {
            steps {
                echo 'Testing'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying'
            }
        }
    }
}
```