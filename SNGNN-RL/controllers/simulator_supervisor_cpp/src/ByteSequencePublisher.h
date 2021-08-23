// **********************************************************************
//
// Copyright (c) 2003-2017 ZeroC, Inc. All rights reserved.
//
// This copy of Ice is licensed to you under the terms described in the
// ICE_LICENSE file included in this distribution.
//
// **********************************************************************
//
// Ice version 3.7.0
//
// <auto-generated>
//
// Generated from file `ByteSequencePublisher.ice'
//
// Warning: do not edit this file.
//
// </auto-generated>
//

#ifndef __ByteSequencePublisher_h__
#define __ByteSequencePublisher_h__

#include <IceUtil/PushDisableWarnings.h>
#include <Ice/ProxyF.h>
#include <Ice/ObjectF.h>
#include <Ice/ValueF.h>
#include <Ice/Exception.h>
#include <Ice/LocalObject.h>
#include <Ice/StreamHelpers.h>
#include <Ice/Comparable.h>
#include <Ice/Proxy.h>
#include <Ice/Object.h>
#include <Ice/GCObject.h>
#include <Ice/Value.h>
#include <Ice/Incoming.h>
#include <Ice/FactoryTableInit.h>
#include <IceUtil/ScopedArray.h>
#include <Ice/Optional.h>
#include <IceUtil/UndefSysMacros.h>

#ifndef ICE_IGNORE_VERSION
#   if ICE_INT_VERSION / 100 != 307
#       error Ice version mismatch!
#   endif
#   if ICE_INT_VERSION % 100 > 50
#       error Beta header file detected
#   endif
#   if ICE_INT_VERSION % 100 < 0
#       error Ice patch level mismatch!
#   endif
#endif

#ifdef ICE_CPP11_MAPPING // C++11 mapping

namespace RoboCompByteSequencePublisher
{

class ByteSequencePublisher;
class ByteSequencePublisherPrx;

}

namespace RoboCompByteSequencePublisher
{

using bytesequence = ::std::vector<::Ice::Byte>;

}

namespace RoboCompByteSequencePublisher
{

class ByteSequencePublisher : public virtual ::Ice::Object
{
public:

    using ProxyType = ByteSequencePublisherPrx;

    virtual bool ice_isA(::std::string, const ::Ice::Current&) const override;
    virtual ::std::vector<::std::string> ice_ids(const ::Ice::Current&) const override;
    virtual ::std::string ice_id(const ::Ice::Current&) const override;

    static const ::std::string& ice_staticId();

    virtual void newsequence(::RoboCompByteSequencePublisher::bytesequence, const ::Ice::Current&) = 0;
    bool _iceD_newsequence(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual bool _iceDispatch(::IceInternal::Incoming&, const ::Ice::Current&) override;
};

}

namespace RoboCompByteSequencePublisher
{

class ByteSequencePublisherPrx : public virtual ::Ice::Proxy<ByteSequencePublisherPrx, ::Ice::ObjectPrx>
{
public:

    void newsequence(const ::RoboCompByteSequencePublisher::bytesequence& iceP_bs, const ::Ice::Context& context = Ice::noExplicitContext)
    {
        _makePromiseOutgoing<void>(true, this, &RoboCompByteSequencePublisher::ByteSequencePublisherPrx::_iceI_newsequence, iceP_bs, context).get();
    }

    template<template<typename> class P = ::std::promise>
    auto newsequenceAsync(const ::RoboCompByteSequencePublisher::bytesequence& iceP_bs, const ::Ice::Context& context = Ice::noExplicitContext)
        -> decltype(::std::declval<P<void>>().get_future())
    {
        return _makePromiseOutgoing<void, P>(false, this, &RoboCompByteSequencePublisher::ByteSequencePublisherPrx::_iceI_newsequence, iceP_bs, context);
    }

    ::std::function<void()>
    newsequenceAsync(const ::RoboCompByteSequencePublisher::bytesequence& iceP_bs,
                     ::std::function<void()> response,
                     ::std::function<void(::std::exception_ptr)> ex = nullptr,
                     ::std::function<void(bool)> sent = nullptr,
                     const ::Ice::Context& context = Ice::noExplicitContext)
    {
        return _makeLamdaOutgoing<void>(response, ex, sent, this, &RoboCompByteSequencePublisher::ByteSequencePublisherPrx::_iceI_newsequence, iceP_bs, context);
    }

    void _iceI_newsequence(const ::std::shared_ptr<::IceInternal::OutgoingAsyncT<void>>&, const ::RoboCompByteSequencePublisher::bytesequence&, const ::Ice::Context&);

    static const ::std::string& ice_staticId();

protected:

    ByteSequencePublisherPrx() = default;
    friend ::std::shared_ptr<ByteSequencePublisherPrx> IceInternal::createProxy<ByteSequencePublisherPrx>();

    virtual ::std::shared_ptr<::Ice::ObjectPrx> _newInstance() const override;
};

}

namespace Ice
{

}

namespace RoboCompByteSequencePublisher
{

using ByteSequencePublisherPtr = ::std::shared_ptr<ByteSequencePublisher>;
using ByteSequencePublisherPrxPtr = ::std::shared_ptr<ByteSequencePublisherPrx>;

}

#else // C++98 mapping

namespace IceProxy
{

namespace RoboCompByteSequencePublisher
{

class ByteSequencePublisher;
void _readProxy(::Ice::InputStream*, ::IceInternal::ProxyHandle< ::IceProxy::RoboCompByteSequencePublisher::ByteSequencePublisher>&);
::IceProxy::Ice::Object* upCast(::IceProxy::RoboCompByteSequencePublisher::ByteSequencePublisher*);

}

}

namespace RoboCompByteSequencePublisher
{

class ByteSequencePublisher;
::Ice::Object* upCast(::RoboCompByteSequencePublisher::ByteSequencePublisher*);
typedef ::IceInternal::Handle< ::RoboCompByteSequencePublisher::ByteSequencePublisher> ByteSequencePublisherPtr;
typedef ::IceInternal::ProxyHandle< ::IceProxy::RoboCompByteSequencePublisher::ByteSequencePublisher> ByteSequencePublisherPrx;
typedef ByteSequencePublisherPrx ByteSequencePublisherPrxPtr;
void _icePatchObjectPtr(ByteSequencePublisherPtr&, const ::Ice::ObjectPtr&);

}

namespace RoboCompByteSequencePublisher
{

typedef ::std::vector< ::Ice::Byte> bytesequence;

}

namespace RoboCompByteSequencePublisher
{

class Callback_ByteSequencePublisher_newsequence_Base : public virtual ::IceInternal::CallbackBase { };
typedef ::IceUtil::Handle< Callback_ByteSequencePublisher_newsequence_Base> Callback_ByteSequencePublisher_newsequencePtr;

}

namespace IceProxy
{

namespace RoboCompByteSequencePublisher
{

class ByteSequencePublisher : public virtual ::Ice::Proxy<ByteSequencePublisher, ::IceProxy::Ice::Object>
{
public:

    void newsequence(const ::RoboCompByteSequencePublisher::bytesequence& iceP_bs, const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        end_newsequence(_iceI_begin_newsequence(iceP_bs, context, ::IceInternal::dummyCallback, 0, true));
    }

    ::Ice::AsyncResultPtr begin_newsequence(const ::RoboCompByteSequencePublisher::bytesequence& iceP_bs, const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        return _iceI_begin_newsequence(iceP_bs, context, ::IceInternal::dummyCallback, 0);
    }

    ::Ice::AsyncResultPtr begin_newsequence(const ::RoboCompByteSequencePublisher::bytesequence& iceP_bs, const ::Ice::CallbackPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_newsequence(iceP_bs, ::Ice::noExplicitContext, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_newsequence(const ::RoboCompByteSequencePublisher::bytesequence& iceP_bs, const ::Ice::Context& context, const ::Ice::CallbackPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_newsequence(iceP_bs, context, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_newsequence(const ::RoboCompByteSequencePublisher::bytesequence& iceP_bs, const ::RoboCompByteSequencePublisher::Callback_ByteSequencePublisher_newsequencePtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_newsequence(iceP_bs, ::Ice::noExplicitContext, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_newsequence(const ::RoboCompByteSequencePublisher::bytesequence& iceP_bs, const ::Ice::Context& context, const ::RoboCompByteSequencePublisher::Callback_ByteSequencePublisher_newsequencePtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_newsequence(iceP_bs, context, del, cookie);
    }

    void end_newsequence(const ::Ice::AsyncResultPtr&);

private:

    ::Ice::AsyncResultPtr _iceI_begin_newsequence(const ::RoboCompByteSequencePublisher::bytesequence&, const ::Ice::Context&, const ::IceInternal::CallbackBasePtr&, const ::Ice::LocalObjectPtr& cookie = 0, bool sync = false);

public:

    static const ::std::string& ice_staticId();

protected:

    virtual ::IceProxy::Ice::Object* _newInstance() const;
};

}

}

namespace RoboCompByteSequencePublisher
{

class ByteSequencePublisher : public virtual ::Ice::Object
{
public:

    typedef ByteSequencePublisherPrx ProxyType;
    typedef ByteSequencePublisherPtr PointerType;

    virtual ~ByteSequencePublisher();

    virtual bool ice_isA(const ::std::string&, const ::Ice::Current& = ::Ice::emptyCurrent) const;
    virtual ::std::vector< ::std::string> ice_ids(const ::Ice::Current& = ::Ice::emptyCurrent) const;
    virtual const ::std::string& ice_id(const ::Ice::Current& = ::Ice::emptyCurrent) const;

    static const ::std::string& ice_staticId();

    virtual void newsequence(const ::RoboCompByteSequencePublisher::bytesequence&, const ::Ice::Current& = ::Ice::emptyCurrent) = 0;
    bool _iceD_newsequence(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual bool _iceDispatch(::IceInternal::Incoming&, const ::Ice::Current&);

protected:

    virtual void _iceWriteImpl(::Ice::OutputStream*) const;
    virtual void _iceReadImpl(::Ice::InputStream*);
};

inline bool operator==(const ByteSequencePublisher& lhs, const ByteSequencePublisher& rhs)
{
    return static_cast<const ::Ice::Object&>(lhs) == static_cast<const ::Ice::Object&>(rhs);
}

inline bool operator<(const ByteSequencePublisher& lhs, const ByteSequencePublisher& rhs)
{
    return static_cast<const ::Ice::Object&>(lhs) < static_cast<const ::Ice::Object&>(rhs);
}

}

namespace Ice
{

}

namespace RoboCompByteSequencePublisher
{

template<class T>
class CallbackNC_ByteSequencePublisher_newsequence : public Callback_ByteSequencePublisher_newsequence_Base, public ::IceInternal::OnewayCallbackNC<T>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception&);
    typedef void (T::*Sent)(bool);
    typedef void (T::*Response)();

    CallbackNC_ByteSequencePublisher_newsequence(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::OnewayCallbackNC<T>(obj, cb, excb, sentcb)
    {
    }
};

template<class T> Callback_ByteSequencePublisher_newsequencePtr
newCallback_ByteSequencePublisher_newsequence(const IceUtil::Handle<T>& instance, void (T::*cb)(), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_ByteSequencePublisher_newsequence<T>(instance, cb, excb, sentcb);
}

template<class T> Callback_ByteSequencePublisher_newsequencePtr
newCallback_ByteSequencePublisher_newsequence(const IceUtil::Handle<T>& instance, void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_ByteSequencePublisher_newsequence<T>(instance, 0, excb, sentcb);
}

template<class T> Callback_ByteSequencePublisher_newsequencePtr
newCallback_ByteSequencePublisher_newsequence(T* instance, void (T::*cb)(), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_ByteSequencePublisher_newsequence<T>(instance, cb, excb, sentcb);
}

template<class T> Callback_ByteSequencePublisher_newsequencePtr
newCallback_ByteSequencePublisher_newsequence(T* instance, void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_ByteSequencePublisher_newsequence<T>(instance, 0, excb, sentcb);
}

template<class T, typename CT>
class Callback_ByteSequencePublisher_newsequence : public Callback_ByteSequencePublisher_newsequence_Base, public ::IceInternal::OnewayCallback<T, CT>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception& , const CT&);
    typedef void (T::*Sent)(bool , const CT&);
    typedef void (T::*Response)(const CT&);

    Callback_ByteSequencePublisher_newsequence(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::OnewayCallback<T, CT>(obj, cb, excb, sentcb)
    {
    }
};

template<class T, typename CT> Callback_ByteSequencePublisher_newsequencePtr
newCallback_ByteSequencePublisher_newsequence(const IceUtil::Handle<T>& instance, void (T::*cb)(const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_ByteSequencePublisher_newsequence<T, CT>(instance, cb, excb, sentcb);
}

template<class T, typename CT> Callback_ByteSequencePublisher_newsequencePtr
newCallback_ByteSequencePublisher_newsequence(const IceUtil::Handle<T>& instance, void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_ByteSequencePublisher_newsequence<T, CT>(instance, 0, excb, sentcb);
}

template<class T, typename CT> Callback_ByteSequencePublisher_newsequencePtr
newCallback_ByteSequencePublisher_newsequence(T* instance, void (T::*cb)(const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_ByteSequencePublisher_newsequence<T, CT>(instance, cb, excb, sentcb);
}

template<class T, typename CT> Callback_ByteSequencePublisher_newsequencePtr
newCallback_ByteSequencePublisher_newsequence(T* instance, void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_ByteSequencePublisher_newsequence<T, CT>(instance, 0, excb, sentcb);
}

}

#endif

#include <IceUtil/PopDisableWarnings.h>
#endif