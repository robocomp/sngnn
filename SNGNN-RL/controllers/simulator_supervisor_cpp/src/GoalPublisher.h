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
// Generated from file `GoalPublisher.ice'
//
// Warning: do not edit this file.
//
// </auto-generated>
//

#ifndef __GoalPublisher_h__
#define __GoalPublisher_h__

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

namespace RoboCompGoalPublisher
{

class GoalPublisher;
class GoalPublisherPrx;

}

namespace RoboCompGoalPublisher
{

struct GoalT
{
    float timestamp;
    float x;
    float y;

    std::tuple<const float&, const float&, const float&> ice_tuple() const
    {
        return std::tie(timestamp, x, y);
    }
};

using Ice::operator<;
using Ice::operator<=;
using Ice::operator>;
using Ice::operator>=;
using Ice::operator==;
using Ice::operator!=;

}

namespace RoboCompGoalPublisher
{

class GoalPublisher : public virtual ::Ice::Object
{
public:

    using ProxyType = GoalPublisherPrx;

    virtual bool ice_isA(::std::string, const ::Ice::Current&) const override;
    virtual ::std::vector<::std::string> ice_ids(const ::Ice::Current&) const override;
    virtual ::std::string ice_id(const ::Ice::Current&) const override;

    static const ::std::string& ice_staticId();

    virtual void goalupdated(::RoboCompGoalPublisher::GoalT, const ::Ice::Current&) = 0;
    bool _iceD_goalupdated(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual bool _iceDispatch(::IceInternal::Incoming&, const ::Ice::Current&) override;
};

}

namespace RoboCompGoalPublisher
{

class GoalPublisherPrx : public virtual ::Ice::Proxy<GoalPublisherPrx, ::Ice::ObjectPrx>
{
public:

    void goalupdated(const ::RoboCompGoalPublisher::GoalT& iceP_goal, const ::Ice::Context& context = Ice::noExplicitContext)
    {
        _makePromiseOutgoing<void>(true, this, &RoboCompGoalPublisher::GoalPublisherPrx::_iceI_goalupdated, iceP_goal, context).get();
    }

    template<template<typename> class P = ::std::promise>
    auto goalupdatedAsync(const ::RoboCompGoalPublisher::GoalT& iceP_goal, const ::Ice::Context& context = Ice::noExplicitContext)
        -> decltype(::std::declval<P<void>>().get_future())
    {
        return _makePromiseOutgoing<void, P>(false, this, &RoboCompGoalPublisher::GoalPublisherPrx::_iceI_goalupdated, iceP_goal, context);
    }

    ::std::function<void()>
    goalupdatedAsync(const ::RoboCompGoalPublisher::GoalT& iceP_goal,
                     ::std::function<void()> response,
                     ::std::function<void(::std::exception_ptr)> ex = nullptr,
                     ::std::function<void(bool)> sent = nullptr,
                     const ::Ice::Context& context = Ice::noExplicitContext)
    {
        return _makeLamdaOutgoing<void>(response, ex, sent, this, &RoboCompGoalPublisher::GoalPublisherPrx::_iceI_goalupdated, iceP_goal, context);
    }

    void _iceI_goalupdated(const ::std::shared_ptr<::IceInternal::OutgoingAsyncT<void>>&, const ::RoboCompGoalPublisher::GoalT&, const ::Ice::Context&);

    static const ::std::string& ice_staticId();

protected:

    GoalPublisherPrx() = default;
    friend ::std::shared_ptr<GoalPublisherPrx> IceInternal::createProxy<GoalPublisherPrx>();

    virtual ::std::shared_ptr<::Ice::ObjectPrx> _newInstance() const override;
};

}

namespace Ice
{

template<>
struct StreamableTraits<::RoboCompGoalPublisher::GoalT>
{
    static const StreamHelperCategory helper = StreamHelperCategoryStruct;
    static const int minWireSize = 12;
    static const bool fixedLength = true;
};

template<typename S>
struct StreamReader<::RoboCompGoalPublisher::GoalT, S>
{
    static void read(S* istr, ::RoboCompGoalPublisher::GoalT& v)
    {
        istr->readAll(v.timestamp, v.x, v.y);
    }
};

}

namespace RoboCompGoalPublisher
{

using GoalPublisherPtr = ::std::shared_ptr<GoalPublisher>;
using GoalPublisherPrxPtr = ::std::shared_ptr<GoalPublisherPrx>;

}

#else // C++98 mapping

namespace IceProxy
{

namespace RoboCompGoalPublisher
{

class GoalPublisher;
void _readProxy(::Ice::InputStream*, ::IceInternal::ProxyHandle< ::IceProxy::RoboCompGoalPublisher::GoalPublisher>&);
::IceProxy::Ice::Object* upCast(::IceProxy::RoboCompGoalPublisher::GoalPublisher*);

}

}

namespace RoboCompGoalPublisher
{

class GoalPublisher;
::Ice::Object* upCast(::RoboCompGoalPublisher::GoalPublisher*);
typedef ::IceInternal::Handle< ::RoboCompGoalPublisher::GoalPublisher> GoalPublisherPtr;
typedef ::IceInternal::ProxyHandle< ::IceProxy::RoboCompGoalPublisher::GoalPublisher> GoalPublisherPrx;
typedef GoalPublisherPrx GoalPublisherPrxPtr;
void _icePatchObjectPtr(GoalPublisherPtr&, const ::Ice::ObjectPtr&);

}

namespace RoboCompGoalPublisher
{

struct GoalT
{
    ::Ice::Float timestamp;
    ::Ice::Float x;
    ::Ice::Float y;
};

}

namespace RoboCompGoalPublisher
{

class Callback_GoalPublisher_goalupdated_Base : public virtual ::IceInternal::CallbackBase { };
typedef ::IceUtil::Handle< Callback_GoalPublisher_goalupdated_Base> Callback_GoalPublisher_goalupdatedPtr;

}

namespace IceProxy
{

namespace RoboCompGoalPublisher
{

class GoalPublisher : public virtual ::Ice::Proxy<GoalPublisher, ::IceProxy::Ice::Object>
{
public:

    void goalupdated(const ::RoboCompGoalPublisher::GoalT& iceP_goal, const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        end_goalupdated(_iceI_begin_goalupdated(iceP_goal, context, ::IceInternal::dummyCallback, 0, true));
    }

    ::Ice::AsyncResultPtr begin_goalupdated(const ::RoboCompGoalPublisher::GoalT& iceP_goal, const ::Ice::Context& context = ::Ice::noExplicitContext)
    {
        return _iceI_begin_goalupdated(iceP_goal, context, ::IceInternal::dummyCallback, 0);
    }

    ::Ice::AsyncResultPtr begin_goalupdated(const ::RoboCompGoalPublisher::GoalT& iceP_goal, const ::Ice::CallbackPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_goalupdated(iceP_goal, ::Ice::noExplicitContext, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_goalupdated(const ::RoboCompGoalPublisher::GoalT& iceP_goal, const ::Ice::Context& context, const ::Ice::CallbackPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_goalupdated(iceP_goal, context, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_goalupdated(const ::RoboCompGoalPublisher::GoalT& iceP_goal, const ::RoboCompGoalPublisher::Callback_GoalPublisher_goalupdatedPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_goalupdated(iceP_goal, ::Ice::noExplicitContext, del, cookie);
    }

    ::Ice::AsyncResultPtr begin_goalupdated(const ::RoboCompGoalPublisher::GoalT& iceP_goal, const ::Ice::Context& context, const ::RoboCompGoalPublisher::Callback_GoalPublisher_goalupdatedPtr& del, const ::Ice::LocalObjectPtr& cookie = 0)
    {
        return _iceI_begin_goalupdated(iceP_goal, context, del, cookie);
    }

    void end_goalupdated(const ::Ice::AsyncResultPtr&);

private:

    ::Ice::AsyncResultPtr _iceI_begin_goalupdated(const ::RoboCompGoalPublisher::GoalT&, const ::Ice::Context&, const ::IceInternal::CallbackBasePtr&, const ::Ice::LocalObjectPtr& cookie = 0, bool sync = false);

public:

    static const ::std::string& ice_staticId();

protected:

    virtual ::IceProxy::Ice::Object* _newInstance() const;
};

}

}

namespace RoboCompGoalPublisher
{

class GoalPublisher : public virtual ::Ice::Object
{
public:

    typedef GoalPublisherPrx ProxyType;
    typedef GoalPublisherPtr PointerType;

    virtual ~GoalPublisher();

    virtual bool ice_isA(const ::std::string&, const ::Ice::Current& = ::Ice::emptyCurrent) const;
    virtual ::std::vector< ::std::string> ice_ids(const ::Ice::Current& = ::Ice::emptyCurrent) const;
    virtual const ::std::string& ice_id(const ::Ice::Current& = ::Ice::emptyCurrent) const;

    static const ::std::string& ice_staticId();

    virtual void goalupdated(const ::RoboCompGoalPublisher::GoalT&, const ::Ice::Current& = ::Ice::emptyCurrent) = 0;
    bool _iceD_goalupdated(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual bool _iceDispatch(::IceInternal::Incoming&, const ::Ice::Current&);

protected:

    virtual void _iceWriteImpl(::Ice::OutputStream*) const;
    virtual void _iceReadImpl(::Ice::InputStream*);
};

inline bool operator==(const GoalPublisher& lhs, const GoalPublisher& rhs)
{
    return static_cast<const ::Ice::Object&>(lhs) == static_cast<const ::Ice::Object&>(rhs);
}

inline bool operator<(const GoalPublisher& lhs, const GoalPublisher& rhs)
{
    return static_cast<const ::Ice::Object&>(lhs) < static_cast<const ::Ice::Object&>(rhs);
}

}

namespace Ice
{

template<>
struct StreamableTraits< ::RoboCompGoalPublisher::GoalT>
{
    static const StreamHelperCategory helper = StreamHelperCategoryStruct;
    static const int minWireSize = 12;
    static const bool fixedLength = true;
};

template<typename S>
struct StreamWriter< ::RoboCompGoalPublisher::GoalT, S>
{
    static void write(S* ostr, const ::RoboCompGoalPublisher::GoalT& v)
    {
        ostr->write(v.timestamp);
        ostr->write(v.x);
        ostr->write(v.y);
    }
};

template<typename S>
struct StreamReader< ::RoboCompGoalPublisher::GoalT, S>
{
    static void read(S* istr, ::RoboCompGoalPublisher::GoalT& v)
    {
        istr->read(v.timestamp);
        istr->read(v.x);
        istr->read(v.y);
    }
};

}

namespace RoboCompGoalPublisher
{

template<class T>
class CallbackNC_GoalPublisher_goalupdated : public Callback_GoalPublisher_goalupdated_Base, public ::IceInternal::OnewayCallbackNC<T>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception&);
    typedef void (T::*Sent)(bool);
    typedef void (T::*Response)();

    CallbackNC_GoalPublisher_goalupdated(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::OnewayCallbackNC<T>(obj, cb, excb, sentcb)
    {
    }
};

template<class T> Callback_GoalPublisher_goalupdatedPtr
newCallback_GoalPublisher_goalupdated(const IceUtil::Handle<T>& instance, void (T::*cb)(), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_GoalPublisher_goalupdated<T>(instance, cb, excb, sentcb);
}

template<class T> Callback_GoalPublisher_goalupdatedPtr
newCallback_GoalPublisher_goalupdated(const IceUtil::Handle<T>& instance, void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_GoalPublisher_goalupdated<T>(instance, 0, excb, sentcb);
}

template<class T> Callback_GoalPublisher_goalupdatedPtr
newCallback_GoalPublisher_goalupdated(T* instance, void (T::*cb)(), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_GoalPublisher_goalupdated<T>(instance, cb, excb, sentcb);
}

template<class T> Callback_GoalPublisher_goalupdatedPtr
newCallback_GoalPublisher_goalupdated(T* instance, void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_GoalPublisher_goalupdated<T>(instance, 0, excb, sentcb);
}

template<class T, typename CT>
class Callback_GoalPublisher_goalupdated : public Callback_GoalPublisher_goalupdated_Base, public ::IceInternal::OnewayCallback<T, CT>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception& , const CT&);
    typedef void (T::*Sent)(bool , const CT&);
    typedef void (T::*Response)(const CT&);

    Callback_GoalPublisher_goalupdated(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::OnewayCallback<T, CT>(obj, cb, excb, sentcb)
    {
    }
};

template<class T, typename CT> Callback_GoalPublisher_goalupdatedPtr
newCallback_GoalPublisher_goalupdated(const IceUtil::Handle<T>& instance, void (T::*cb)(const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_GoalPublisher_goalupdated<T, CT>(instance, cb, excb, sentcb);
}

template<class T, typename CT> Callback_GoalPublisher_goalupdatedPtr
newCallback_GoalPublisher_goalupdated(const IceUtil::Handle<T>& instance, void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_GoalPublisher_goalupdated<T, CT>(instance, 0, excb, sentcb);
}

template<class T, typename CT> Callback_GoalPublisher_goalupdatedPtr
newCallback_GoalPublisher_goalupdated(T* instance, void (T::*cb)(const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_GoalPublisher_goalupdated<T, CT>(instance, cb, excb, sentcb);
}

template<class T, typename CT> Callback_GoalPublisher_goalupdatedPtr
newCallback_GoalPublisher_goalupdated(T* instance, void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_GoalPublisher_goalupdated<T, CT>(instance, 0, excb, sentcb);
}

}

#endif

#include <IceUtil/PopDisableWarnings.h>
#endif