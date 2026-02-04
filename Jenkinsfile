pipeline {
  agent any

  options { timestamps() }

  environment {
    // Docker Hub repo (must exist under your Docker Hub username)
    IMAGE_REPO = "arvind2733/arvmldevopspipeline"

    // Kubernetes identity
    NAMESPACE  = "arvmldevopspipeline"
    APP_NAME   = "arvmldevopspipeline"        // Deployment name
    CONTAINER  = "arvmldevopspipeline"        // Container name in Deployment spec
    SERVICE    = "arvmldevopspipeline-svc"    // Service name

    // Paths in repo
    K8S_DIR    = "k8s"
    K6_SCRIPT  = "loadtest\\k6.js"
  }

  stages {

    stage('Checkout') {
      steps {
        bat '''
          @echo on
          git --version
          docker --version
          kubectl version --client=true
          git status
          git rev-parse HEAD
        '''
      }
    }

    stage('Compute Tags') {
      steps {
        script {
          env.SHORTSHA = bat(returnStdout: true, script: '@echo off\r\ngit rev-parse --short=7 HEAD').trim()

          def s = bat(returnStatus: true, script: '@echo off\r\ngit describe --tags --exact-match 1> .reltag.txt 2>nul')
          env.RELTAG = (s == 0) ? readFile('.reltag.txt').trim() : ''
        }

        bat '''
          @echo on
          echo SHORTSHA=%SHORTSHA%
          echo RELTAG=%RELTAG%
        '''
      }
    }

    stage('QA (Unit Tests)') {
  steps {
    bat '''
      @echo on
      if not exist reports mkdir reports

      docker build -t %IMAGE_REPO%:test-%SHORTSHA% -f Dockerfile.test .

      docker create --name testrun %IMAGE_REPO%:test-%SHORTSHA%
      docker start -a testrun

      docker cp testrun:/app/reports/test-results.xml "%WORKSPACE%\\reports\\test-results.xml"
      docker rm -f testrun
    '''
  }
  post {
    always {
      junit 'reports/test-results.xml'
      archiveArtifacts artifacts: 'reports/test-results.xml', fingerprint: true
      bat 'docker image rm -f %IMAGE_REPO%:test-%SHORTSHA% 2>nul || exit /b 0'
    }
  }
}

    stage('Build Image') {
      steps {
        script {
          if (!(env.SHORTSHA ==~ /^[0-9a-f]{7}$/)) {
            error("SHORTSHA invalid. Expected 7-hex, got: '${env.SHORTSHA}'")
          }
          if (env.RELTAG && !(env.RELTAG ==~ /^[0-9A-Za-z._-]+$/)) {
            error("RELTAG invalid. Got: '${env.RELTAG}'")
          }
        }

        bat '''
          @echo on
          echo Building image: "%IMAGE_REPO%:git-%SHORTSHA%"
          docker build -t "%IMAGE_REPO%:git-%SHORTSHA%" .

          if not "%RELTAG%"=="" (
            echo Applying release tag: %RELTAG%
            docker tag "%IMAGE_REPO%:git-%SHORTSHA%" "%IMAGE_REPO%:%RELTAG%"
          ) else (
            echo No release tag. Skipping SemVer tag.
          )
        '''
      }
    }

    stage('Push Image') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'dockerhub-creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
          bat '''
            @echo on
            echo %DOCKER_PASS% | docker login -u %DOCKER_USER% --password-stdin

            docker push %IMAGE_REPO%:git-%SHORTSHA%

            if not "%RELTAG%"=="" (
              docker push %IMAGE_REPO%:%RELTAG%
            ) else (
              echo No release tag. Skipping push for SemVer tag.
            )
          '''
        }
      }
    }

    stage('Deploy to Kubernetes (Apply + Set Image)') {
      steps {
        withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG_FILE')]) {
          bat '''
            @echo on
            set KUBECONFIG=%KUBECONFIG_FILE%

            if not exist "%K8S_DIR%\\namespace.yaml" exit /b 1
            if not exist "%K8S_DIR%\\deployment.yaml" exit /b 1
            if not exist "%K8S_DIR%\\service.yaml" exit /b 1

            kubectl apply -f %K8S_DIR%\\namespace.yaml
            kubectl apply -f %K8S_DIR%\\deployment.yaml
            kubectl apply -f %K8S_DIR%\\service.yaml

            echo Setting image to: %IMAGE_REPO%:git-%SHORTSHA%
            kubectl -n %NAMESPACE% set image deployment/%APP_NAME% %CONTAINER%=%IMAGE_REPO%:git-%SHORTSHA%
          '''
        }
      }
    }

    stage('Rollout Gate (Must Succeed)') {
      steps {
        withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG_FILE')]) {
          bat '''
            @echo on
            set KUBECONFIG=%KUBECONFIG_FILE%

            kubectl -n %NAMESPACE% rollout status deployment/%APP_NAME% --timeout=180s
            kubectl -n %NAMESPACE% get pods -o wide
            kubectl -n %NAMESPACE% get svc
          '''
        }
      }
    }
stage('Smoke Test (/health + /predict)') {
  steps {
    withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG_FILE')]) {
      bat """
        set KUBECONFIG=%KUBECONFIG_FILE%

        echo ===== smoke test /health =====
        kubectl -n arvmldevopspipeline run curl-81 --rm -i --restart=Never --image=curlimages/curl -- ^
          curl -f -sS --max-time 10 http://arvmldevopspipeline-svc:8000/health
        if errorlevel 1 exit /b 1

        echo ===== smoke test /predict =====
        (echo {^
"event_ts":"2026-02-03T00:00:00Z",^
"baseline_queue_min":1.0,^
"shortage_flag":0,^
"replenishment_eta_min":5.0,^
"machine_state":"RUN",^
"queue_time_min":2.0,^
"down_minutes_last_60":0.0^
}) > payload.json

        kubectl -n arvmldevopspipeline run curl-81 --rm -i --restart=Never --image=curlimages/curl -- sh -lc ^
          "curl -f -sS --max-time 10 -H \\"Content-Type: application/json\\" -X POST http://arvmldevopspipeline-svc:8000/predict --data-binary @payload.json"
        if errorlevel 1 exit /b 1
      """
    }
  }
}





    stage('Load Test (k6)') {
      steps {
        withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG_FILE')]) {
          bat '''
            @echo on
            set KUBECONFIG=%KUBECONFIG_FILE%

            set NS=%NAMESPACE%
            set JOB=k6
            set CM=k6-script
            set SCRIPT=%WORKSPACE%\\%K6_SCRIPT%

            if not exist "%SCRIPT%" (
              echo ERROR: k6 script not found at path: %SCRIPT%
              dir /s "%WORKSPACE%\\loadtest"
              exit /b 1
            )

            echo ===== k6: cleanup old resources =====
            kubectl -n %NS% delete job %JOB% --ignore-not-found
            kubectl -n %NS% delete configmap %CM% --ignore-not-found

            echo ===== k6: create configmap from script =====
            kubectl -n %NS% create configmap %CM% --from-file=k6.js="%SCRIPT%" || exit /b 1

            echo ===== validate service/endpoints =====
            kubectl -n %NS% get svc %SERVICE% -o wide || exit /b 1
            kubectl -n %NS% get endpoints %SERVICE% -o wide || exit /b 1

            echo ===== k6: create job manifest =====
            (
              echo apiVersion: batch/v1
              echo kind: Job
              echo metadata:
              echo   name: %JOB%
              echo   namespace: %NS%
              echo spec:
              echo   backoffLimit: 0
              echo   template:
              echo     metadata:
              echo       labels:
              echo         app: k6
              echo     spec:
              echo       restartPolicy: Never
              echo       containers:
              echo       - name: k6
              echo         image: grafana/k6:0.48.0
              echo         env:
              echo         - name: BASE_URL
              echo           value: "http://%SERVICE%:8000"
              echo         volumeMounts:
              echo         - name: script
              echo           mountPath: /scripts
              echo         command: ["k6","run","/scripts/k6.js"]
              echo       volumes:
              echo       - name: script
              echo         configMap:
              echo           name: %CM%
            ) > k6-job.yaml

            kubectl apply -f k6-job.yaml || exit /b 1

            echo ===== k6: wait for completion (5 min) =====
            kubectl -n %NS% wait --for=condition=complete job/%JOB% --timeout=300s
            set RC=%ERRORLEVEL%

            if not "%RC%"=="0" (
              echo ===== k6: FAILED - debugging =====
              kubectl -n %NS% describe job/%JOB%
              kubectl -n %NS% logs -l app=k6 --tail=-1
              exit /b 1
            )

            echo ===== k6: SUCCESS - final logs =====
            kubectl -n %NS% logs -l app=k6 --tail=-1
          '''
        }
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'k8s/**/*,loadtest/**/*,Dockerfile,Jenkinsfile,requirements.txt,README.md', fingerprint: true
    }
  }
}

