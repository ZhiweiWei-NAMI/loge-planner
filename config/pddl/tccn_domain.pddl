(define (domain tccn-planning)
    (:requirements :strips :typing)

    (:types
        drone location package - object
    )

    (:predicates
        ;; 位置相关
        (at ?drone - drone ?loc - location)
        (package-at ?package - package ?loc - location)
        (on-drone ?package - package ?drone - drone)
        (connected ?loc1 - location ?loc2 - location)
        (available ?loc - location)

        ;; drone状态
        (drone-free ?drone - drone)  ;; drone没有携带任何package

        ;; 任务状态
        (isDelivered ?package - package ?loc - location)  ;; 修改：绑定到具体的package和location
        (isComputed ?loc - location)
        (isCommunicated ?loc - location)
        (isInspected ?loc - location)


    )

    ;; 动作1: MoveToLoc - 无人机移动
    (:action MoveToLoc
        :parameters (?drone - drone ?loc1 - location ?loc2 - location)
        :precondition (and
            (connected ?loc1 ?loc2)
            (at ?drone ?loc1)
            (available ?loc2)
        )
        :effect (and
            (not (at ?drone ?loc1))
            (at ?drone ?loc2)
        )
    )

    ;; 动作2: MoveWithPackage - 无人机携带包裹移动
    (:action MoveWithPackage
        :parameters (?drone - drone ?package - package ?loc1 - location ?loc2 - location)
        :precondition (and
            (connected ?loc1 ?loc2)
            (at ?drone ?loc1)
            (available ?loc2)
            (on-drone ?package ?drone)
            (package-at ?package ?loc1)
        )
        :effect (and
            (not (at ?drone ?loc1))
            (at ?drone ?loc2)
            (not (package-at ?package ?loc1))
            (package-at ?package ?loc2)
        )
    )

    ;; 动作3: PickupPackage - 取货
    (:action PickupPackage
        :parameters (?drone - drone ?package - package ?loc - location)
        :precondition (and
            (at ?drone ?loc)
            (package-at ?package ?loc)
            (drone-free ?drone)  ;; 新增：要求drone没有携带任何package
        )
        :effect (and
            (not (package-at ?package ?loc))
            (on-drone ?package ?drone)
            (not (drone-free ?drone))  ;; drone现在携带了package
        )
    )

    ;; 动作4: HandoverPackage - 交接货物
    (:action HandoverPackage
        :parameters (?drone1 - drone ?drone2 - drone ?package - package ?loc - location)
        :precondition (and
            (at ?drone1 ?loc)
            (at ?drone2 ?loc)
            (on-drone ?package ?drone1)
            (drone-free ?drone2)  ;; 接收package的drone必须是空闲的
        )
        :effect (and
            (not (on-drone ?package ?drone1))
            (on-drone ?package ?drone2)
            (drone-free ?drone1)  ;; drone1交出package后变为空闲
            (not (drone-free ?drone2))  ;; drone2接收package后不再空闲
        )
    )

    ;; 动作5: DeliverPackage - 配送货物
    (:action DeliverPackage
        :parameters (?drone - drone ?package - package ?loc - location)
        :precondition (and
            (at ?drone ?loc)
            (on-drone ?package ?drone)
        )
        :effect (and
            (not (on-drone ?package ?drone))
            (package-at ?package ?loc)
            (isDelivered ?package ?loc)  ;; 修改：绑定到具体的package和location
            (drone-free ?drone)  ;; drone交付后变为空闲状态
        )
    )

    ;; 动作6: AssistCompute - 协助计算
    (:action AssistCompute
        :parameters (?drone - drone ?loc - location)
        :precondition (and
            (at ?drone ?loc)
        )
        :effect (and
            (isComputed ?loc)
        )
    )

    ;; 动作7: AssistCommunicate - 协助通信
    (:action AssistCommunicate
        :parameters (?drone - drone ?loc - location)
        :precondition (and
            (at ?drone ?loc)
        )
        :effect (and
            (isCommunicated ?loc)
        )
    )

    ;; 动作8: Inspect - 巡检
    (:action Inspect
        :parameters (?drone - drone ?loc - location)
        :precondition (and
            (at ?drone ?loc)
        )
        :effect (and
            (isInspected ?loc)
        )
    )
)
