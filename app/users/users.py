from app import login_manager
from app import auth
# from .users import User


class User:
    def __init__(self, localId, idToken, refreshId):
        self.localId = localId
        self.idToken = idToken
        self.refreshId = refreshId

    @property
    def is_active(self):
        return True

    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        try:
            return str(self.refreshId)
        except AttributeError:
            raise NotImplementedError('No `id` attribute - override `get_id`')


@login_manager.user_loader
def load_user(id_):
    try:
        u = auth.refresh(id_)

        userID = u['userId']
        idToken = u['idToken']
        refreshToken = u['refreshToken']

        user_class = User(userID, idToken, refreshToken)

        return user_class

    except Exception:
        return None


@login_manager.request_loader
def load_user_from_request(request):

    # first, try to login using the api_key url arg
    api_key = request.args.get('api_key')
    if api_key:

        user = load_user(api_key)

        if user:
            return user

    # finally, return None if both methods did not login the user
    return None
