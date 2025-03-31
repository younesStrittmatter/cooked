Python-Version: 3.13

## Google Cloud

You'll need to setup a Google Cloud project here: https://console.cloud.google.com/

Login:

```shell
gcloud auth login
```

This will open a browser window and ask you to login with your Google account. After logging in, you will be asked to
give permission to the gcloud command line tool to access your Google account.
If you run into problems with the permissions, sometimes you need to use the application default credentials instead of
the user credentials. This is done with the following command:

```shell
gcloud auth application-default login
```

### Set Project

You need to set the project you want to deploy to. You can do this by running the following command. The project ID can be found in the Google Cloud Console. Make sure to use the full project ID (often has the format `project-name-123456`).
```shell
gcloud config set project <project-id>
```

### Deploy

```shell
gcloud app deploy
```

