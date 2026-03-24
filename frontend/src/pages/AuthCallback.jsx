import React, { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import supabase from '../helper/supabaseClient';
import { getReplitAppUrl } from '../config';

function AuthCallback() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [status, setStatus] = useState('exchanging'); // 'exchanging' | 'setting-session' | 'error'

  useEffect(() => {
    const code = searchParams.get('code');

    if (!code) {
      navigate('/', { replace: true });
      return;
    }

    const replitBase = getReplitAppUrl();
    if (!replitBase) {
      setStatus('error');
      setTimeout(() => navigate('/login', { replace: true, state: { message: 'Session link expired or invalid' } }), 1500);
      return;
    }

    const exchangeUrl = `${replitBase.replace(/\/$/, '')}/api/auth/momentum-exchange?code=${encodeURIComponent(code)}`;

    (async () => {
      try {
        const res = await fetch(exchangeUrl, { method: 'GET' });
        if (!res.ok) {
          setStatus('error');
          setTimeout(() => navigate('/login', { replace: true, state: { message: 'Session link expired or invalid' } }), 1500);
          return;
        }

        const data = await res.json();
        const { access_token, refresh_token } = data || {};

        if (!access_token || !refresh_token) {
          setStatus('error');
          setTimeout(() => navigate('/login', { replace: true, state: { message: 'Session link expired or invalid' } }), 1500);
          return;
        }

        setStatus('setting-session');
        const { error } = await supabase.auth.setSession({ access_token, refresh_token });

        if (error) {
          setStatus('error');
          setTimeout(() => navigate('/login', { replace: true, state: { message: 'Session link expired or invalid' } }), 1500);
          return;
        }

        navigate('/', { replace: true });
      } catch (_) {
        setStatus('error');
        setTimeout(() => navigate('/login', { replace: true, state: { message: 'Session link expired or invalid' } }), 1500);
      }
    })();
  }, [searchParams, navigate]);

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50">
      <div className="text-center">
        {status === 'error' ? (
          <p className="text-gray-600">Redirecting to sign in...</p>
        ) : (
          <p className="text-gray-600">Signing you in...</p>
        )}
      </div>
    </div>
  );
}

export default AuthCallback;
