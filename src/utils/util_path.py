import os


class PathUtils:
    project_name = 'flask_deploy_testing'

    @staticmethod
    def project_path():
        current_full_path = os.path.abspath(os.curdir)
        project_full_path = current_full_path.split(PathUtils.project_name)[0] + PathUtils.project_name
        return project_full_path

    @staticmethod
    def model_path():
        return PathUtils.project_path() + '/models/'
